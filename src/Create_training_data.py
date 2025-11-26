from multiprocessing import Pool
#all created batches are stored in ./Training_Data/ which should exists

def job(pro_id):
    process=str(pro_id)
    from multiprocessing import Pool
    from PIL import Image
    import numpy as np
    import os, umap, torch, h5py, time, sys
    from scipy.ndimage import gaussian_filter
    import h5py
    import random

    torch.manual_seed(pro_id)
    np.random.seed(pro_id)

    from IPython.display import clear_output
    stretch_amplitude=3
    max_displacement=3
    max_sigma=2

    def get_data_birds(diri,start):
        files=os.listdir(diri)
        XSE=[]
        names=[]
        for file in files:
            if '.png' in file:
                L=Image.open(diri+file)
                img=L.convert('L')
                img=img.resize((100,100))
                img=np.array(img)/255
                img[img>0.5]=1
                img[img<0.5]=0
                XSE.append(img)
                names.append(file)
        return XSE,names

    def get_max(img):
        out=np.unravel_index(img.argmax(), img.shape)
        return out

    def get_neighbours(img,py,px,thres):
        points=[[py,px]]
        checker=True
        while checker:
            checker=False
            for p in points:
                posx=p[1]
                posy=p[0]
                for x in [posx-1,posx,posx+1]:
                    for y in [posy-1,posy,posy+1]:
                        try:
                            if img[y][x]>thres: 
                                if [y,x] not in points:
                                    points.append([y,x])
                                    checker=True
                        except: continue
        return np.array(points)

    def clean_image(img,points):
        new_img=img.copy()
        removed_img=np.zeros((100,100),dtype=float)
        for p in points:
            y=p[0]
            x=p[1]
            new_img[y][x]=0
            removed_img[y][x]=1
        return [new_img,removed_img]
    def get_toes(img): 
        toes=[]
        pos=get_max(img)
        points=get_neighbours(img,pos[0],pos[1],0.9)
        cleaner=clean_image(img,points)
        work_img=cleaner[0]
        toe=cleaner[1]
        toes.append(toe)
        for i in range(10):
            if np.amax(work_img)>0.9:
                pos=get_max(work_img)
                points=get_neighbours(img,pos[0],pos[1],0.9)
                cleaner=clean_image(work_img,points)
                work_img=cleaner[0]
                toe=cleaner[1]
                toes.append(toe)
        return toes

    def create_random_toes(toes):
        out=np.zeros((100,100),dtype=float)
        lll=len(toes)
        if lll>1:
            nr=lll-1
        else:
            nr=1
        random.shuffle(toes)
        for i in range(nr):
            out+=toes[i]
        return out


    def aug(img):
        out2=img.copy()
        rot=Image.fromarray(out2)
        rotator=np.random.rand()*60-30
        rot=rot.rotate(rotator, fillcolor=1)
        out2=np.array(rot)
        flipper=np.random.rand()
        if flipper>0.5:
            out2=np.flip(out2,axis=1)
        zoom_max=stretch_amplitude
        zoomY=np.random.randint(-zoom_max,high=zoom_max)
        zoomX=np.random.randint(-zoom_max,high=zoom_max)
        if zoomX>0:
            out2=out2[:,zoomX:100-zoomX]
        if zoomY>0:
            out2=out2[zoomY:100-zoomY,:]
        if zoomX<0:
            add=abs(zoomX)
            new_img=np.ones((len(out2),100+2*add),dtype=float)
            new_img[:,add:100+add]=out2
            out2=new_img
        if zoomY<0:
            add=abs(zoomY)
            new_img=np.ones((100+2*add,len(out2[0])),dtype=float)
            new_img[add:100+add:,]=out2
            out2=new_img   
        if abs(zoomX)>0 or abs(zoomY)>0:
            out2=Image.fromarray(out2)
            out2=out2.resize((100,100))
            out2=np.array(out2)
        out=out2.copy()
        SIGMA=1+np.random.rand()*(max_sigma-1)
        PIX=np.random.randint(1,high=max_displacement)
        for x in range(1,100-PIX,PIX):
            for y in range(1,100-PIX,PIX):
                cur_img=out[y-1:y+1+PIX,x-1:x+1+PIX]
                if np.amin(cur_img)==0 and np.amax(cur_img)==1:
                     out2[y:y+PIX,x:x+PIX]=np.random.randint(0,high=2,size=(PIX,PIX))
        out2=gaussian_filter(out2,sigma=SIGMA)
        THRES=0.4+np.random.rand()*0.2
        out2[out2>THRES]=1
        out2[out2<THRES]=0 
        return out2



    def IMove(L,avg):
        if avg==1:
            Lnew=L
        else:
            Lnew=[sum(L[i:i+avg])/avg for i in range(len(L)-avg)]
        return Lnew


    # Data loading
    dir0='./Ornithopod-train/'
    dir1='./Theropod-train/'
    dir2='./Bird-train/'
    dir3='./Bird-like-train/'
    dir4='./Extant-train/'
    dir5='./Quadrupedal-train/'
    dirq='./BP-train/'
    #we excluded the following for training:
    #'guillaume_2022_14.png','lockley_1995_6.png','lockley_2000b_18.png','olsen_2003_7.png',
    #'xing_2015b_11.png','alcala_2014_1.png','crowell_2021_6.png','lallensack_2016_34b.png',
    #'lockley_2001_17.png','lull_1953_4.png','olsen_2003_24.png','xing_2021g_5.png'
    #'farlow_2018_11.png','farlow_paluxy_2.png','klein_2022_7.png','lallensack_2016_6.png',
    #'lockley_1995_2.png','kim_2017_1.png','lallensack_2016_38.png','lallensack_2022_pizza_overhang_17.png',
    #'lallensack_2022_pizza_rockfall_13.png','lallensack_2022_T3-2015_3.png','leonardi_2021_soca_19.png',
    #'lockley_2000b_6.png','lockley_2013_2.png','reynolds_1989_2.png','Suarez Hernando et al 2016 - Fig 3_1 - 2.png',
    #'Lockley et al 2021 - Fig 2 4 R.png','Sarjeant and Reynolds 2001 - Fig 2 1.png',
    #'Fleury et al 2023 - Fig 3 - R1.png','Sarjeant and Reynolds 2001 - Fig 12 2.png'
    #'Gierlinski et al 2017 - Fig 2b.png','Lockley et al 2007 - Fig 2b.png','Gierlinski et al 2017 - Fig 3b.png',
    #'Elbroch and Mark 2001 - American Robin - 1 R2.png','Elbroch and Mark 2001 - Ring Necked Pheasant 1R.png',
    #'Gibson - Greylag Goose - 3.png','Evans - Great Blue Heron 2.png','Hall et al 2016 - Fig 9_3 - 3.png',
    #'Xing et al 2015 - Fig 10 - LP3.png' and the theropods from Bla25 as well as
    #ornithopod images [Cas22, Día16, Día24, Die04, Enr22, Gar23, Hor16, Lee18, Li23, Loc12, Mat08, 
    #Niu23, Pan24, Pon14, Sar74, Sar98, Shi19, Vil23, Xin21b, Xin25, Xin25b]
    #run this script for the excluded tracks to create validation and test data

    data0,names0=get_data_birds(dir0,'ornithischian')
    data1,names1=get_data_birds(dir1,'theropod')            
    data2,names2=get_data_birds(dir2,'birds')
    data3,names3=get_data_birds(dir3,'birdy')
    data4,names4=get_data_birds(dir4,'ext')
    data5,names5=get_data_birds(dir5,'stompy')
    dataq,namesq=get_data_birds(dirq,'question')


    data0_train=[]
    data1_train=[]
    data2_train=[]
    data3_train=[]
    data4_train=[]
    data5_train=[]


    for i in range(len(data0)):
        data0_train.append(data0[i])

    for i in range(len(data1)):
        data1_train.append(data1[i])   

    for i in range(len(data2)):
        data2_train.append(data2[i])  

    for i in range(len(data3)):
        data3_train.append(data3[i])   

    for i in range(len(data4)):
        data4_train.append(data4[i])   

    for i in range(len(data5)):
        data5_train.append(data5[i])   

    def remove_circle(L):
        xx, yy = np.mgrid[:100, :100]
        cx=np.random.randint(0,high=100)
        cy=np.random.randint(0,high=100)
        r=np.random.rand()*25
        circle = (xx - cx) ** 2 + (yy - cy) ** 2
        donut = np.logical_and(circle < (r**2 + 0), circle > (r**2-6000))
        donut=-np.array(donut,dtype=float)
        out=L+donut
        out[out>0.5]=1
        out[out<0.5]=0
        return out

    def get_train(data0i,data1i,data2i,data3i,data4i,data5i):
        prob0=0.3
        prob1=0.3
        prob2=0.15
        prob3=0.05
        prob4=0.1
        prob5=0.1
        # np.random.seed()
        decider=np.random.rand()
        if decider<prob0:
            data=data0i
        elif decider<prob0+prob1:
            data=data1i
        elif decider<prob0+prob1+prob2:
            data=data2i
        elif decider<prob0+prob1+prob2+prob3:
            data=data3i
        elif decider<prob0+prob1+prob2+prob3+prob4:
            data=data4i
        else: 
            data=data5i
        lll=len(data)
        i=np.random.randint(0,high=lll)
        sample=-aug(data[i])+1
        decider=np.random.rand()
        if decider<0.2:#0.25
            toes=get_toes(sample)
            insample=create_random_toes(toes)
        elif decider<0.4:#0.5
            insample=remove_circle(sample)
        else:
            insample=sample.copy()
        return [insample,sample]


    for i in range(0,150,1):
        x_preload=[]
        idx=int(process)+20*i
        print(idx)
        t0=time.time()
        for j in range(256):
            x_preload.append(get_train(data0_train,data1_train,data2_train,data3_train,data4_train,data5_train))
        t1=time.time()
        print('time needed for 256: ',t1-t0)
        x_preload=np.array(x_preload)

        fx=h5py.File('./Training_Data/batch_'+str(idx)+'.h5','w')
        fx.create_dataset('X',data=x_preload,compression='gzip')
        fx.close()
        clear_output(wait=True)
    print('done!')

Number_Workers=20
Ltodo=range(0,20)
if __name__ == '__main__':
    with Pool(Number_Workers) as p:
        p.map(job, Ltodo)





