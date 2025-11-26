#Pure training, the analysis of the network can be done with the app
#The images and their names are in the images_compressed.npz and names.py

import numpy as np
import os, torch, h5py, time, sys
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn.functional as F
import h5py
import random
torch.manual_seed(0)
np.random.seed(0)

from IPython.display import clear_output


Version_name='BETA15_BIG_3k_shuffle'
IMG_SIZE=100
N_EPOCHS=1000
use_GPU=True

epoch_milestone=100
first_look=10
features=50


filter0,filter1,filter2=60,60,60
dfilter0,dfilter1,dfilter2=40,25,10

start_beta=15
used_beta=start_beta
reach_epoch=100
change_lr=100
downscale_lr=0.5
batch_size=256
learning_rate=1e-8
momentum_rate=0.8
created_batches_number=3000
training_size=created_batches_number*batch_size

PRELOADER=True
Version=Version_name

root_directory='./'
method_export=root_directory+'model_'+str(Version)+'/'

for directory in [method_export]:
    if not os.path.exists(directory):
        print('creating: '+directory)
        os.makedirs(directory)

if use_GPU:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print('start import')
try:
    fx.close()
except:
    print('all files closed')

def set_values(arr):
    arr[arr > 0.5] = 1
    arr[arr <= 0.5] = 0
    return arr

if PRELOADER:
    xpreload=[]
    max_files=created_batches_number
    for idx in range(max_files):
        print('preloading:'+str(idx))
        clear_output(wait=True)
        fx=h5py.File('./Training_Data/batch_'+str(idx)+'.h5','r')
        x_train=np.array(fx['X'])
        for x in x_train:
            x_train2=[[x[0]],[x[1]]]
            x_train2=np.array(x_train2)
            xpreload.append(x_train2)

    def get_train_batch():
        idxs=np.random.randint(0, high=training_size,size=batch_size)
        x_train2=[]
        for idx in idxs:
            x_train2.append(xpreload[idx])
        x_train2=np.array(x_train2)
        x_train=torch.tensor(x_train2,dtype=torch.float, device=device)     
        return x_train
else:
    def get_train_batch():
        idx=np.random.randint(0, high=created_batches_number)
        fx=h5py.File('./Training_Data/batch_'+str(idx)+'.h5','r')
        x_train=np.array(fx['X'])
        fx.close()
        x_train2=[]
        for x in x_train:
            x_train2.append([[x[0]],[x[1]]])
        x_train2=np.array(x_train2)
        x_train=torch.tensor(x_train2,dtype=torch.float, device=device)     
        return x_train

class giveData(Dataset): 
    def __init__(self, x):
        self.x=x
        self.len=len(x)
    def __getitem__(self, index):
        self.xc=self.x[index]
        return self.xc
    def __len__(self):
        return self.len   


len_train=2811*256# historically grown number :-)

conv_size=3
pad=1
stridepool_size=2
pool_size=2

def func_beta(epoch): #scheduled beta parameter
    if epoch<reach_epoch:
        out=start_beta+epoch*(used_beta-start_beta)/reach_epoch
    else:
        out=used_beta
    return out

class mixerVAE(nn.Module):
    def __init__(self):
        super(mixerVAE, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(100, filter0, kernel_size=(conv_size,conv_size), stride=(1,1), padding=(pad,pad)),
            nn.Mish(),
            nn.AvgPool2d(kernel_size=(pool_size,pool_size), stride=(stridepool_size,stridepool_size), padding=(1,1)),
            nn.Conv2d(filter0, filter1, kernel_size=(conv_size,conv_size), stride=(1,1), padding=(pad,pad)),
            nn.Mish(),
            nn.AvgPool2d(kernel_size=(pool_size,pool_size), stride=(stridepool_size,stridepool_size), padding=(1,1)),
            nn.Conv2d(filter1, filter2, kernel_size=(conv_size,conv_size), stride=(1,1), padding=(pad,pad)),
            nn.Mish(),
            nn.AvgPool2d(kernel_size=(pool_size,pool_size), stride=(stridepool_size,stridepool_size), padding=(1,1)),
        )
        self.map_encode=nn.Linear(in_features=filter2*3*3, out_features=2*features)
        self.map_decode=nn.Linear(in_features=features, out_features=50*50)
        self.dec0 = nn.Conv2d(1, dfilter0, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dec1 = nn.Conv2d(dfilter0,dfilter1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dec2 = nn.Conv2d(dfilter1,dfilter2, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dec3 = nn.Conv2d(dfilter2, 4, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x, operation):
        if operation=='x':
            x=x.view(-1,100,10,10)
            x=self.encode(x)
            x=x.view(-1,filter2*3*3)
            x=self.map_encode(x).view(-1, 2, features)
            mu = x[:, 0, :] 
            log_var = x[:, 1, :] 
            z = self.reparameterize(mu, log_var)
            x=self.map_decode(z)
            x=x.view(-1,1,5,5)
            x=F.mish(x)
            x=self.dec0(x)
            x=F.mish(x)
            x=self.dec1(x)
            x=F.mish(x)
            x=self.dec2(x)
            x=F.mish(x)
            x=torch.sigmoid(self.dec3(x))    
            reconstruction = x.view(-1,1,IMG_SIZE,IMG_SIZE)
            return reconstruction,mu,log_var,z

        if operation=='z':
            x=self.map_decode(x)
            x=x.view(-1,1,5,5)
            x=F.mish(x)
            x=self.dec0(x)
            x=F.mish(x)
            x=self.dec1(x)
            x=F.mish(x)
            x=self.dec2(x)
            x=F.mish(x)
            x=torch.sigmoid(self.dec3(x))    
            reconstruction = x.view(-1,1,IMG_SIZE,IMG_SIZE)
            return reconstruction

model = mixerVAE()

optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum_rate)
criterion = nn.MSELoss(reduction='sum')

def final_loss(bce_loss, mu, logvar, BETA):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + BETA*KLD

epoch=0
L_test=[]
L_train=[]
L_beta=[]
L_lr=[]
if use_GPU:
    model.to(device)

# fx=h5py.File('./X_test.h5','r')
# x_test=np.array(fx['X_test'])
# fx.close()
# x_test=torch.tensor(x_test,dtype=torch.float, device=device)   
# test_dataset=giveData(x_test)
# test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=False)

try:
    fe.close()
except:
    print('all files closed')

print('start training...')

for epoch_idx in range(N_EPOCHS+2):   
    model.train()
    t0=time.time()
    epoch+=1
    beta_used=func_beta(epoch)
    if epoch==10:
        learning_rate=1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    if epoch>reach_epoch:
        if (epoch) % change_lr==0:
            learning_rate=learning_rate*downscale_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

    loss_train=0
    L_lr.append(learning_rate)
    L_beta.append(beta_used)

    for idx_counter in range(2811):#historically grown number
        data_train=get_train_batch()
        optimizer.zero_grad()
        reconstruction, mu, logvar,z = model(data_train[:,0],'x')
        bce_loss = criterion(reconstruction, data_train[:,1])
        loss = final_loss(bce_loss, mu, logvar,beta_used)
        loss_train+=loss.item()
        loss.backward()
        optimizer.step()
    loss_train=loss_train/len_train
    t1=time.time()
    # model.eval()
    # loss_test=0
    # reconstruction_test, mu_test, logvar_test,z_test=[],[],[],[]
    # for data_test in test_loader:
    #     reconstruction, mu, logvar,z = model(data_test[:,0],'x')
    #     for i in range(len(reconstruction)):
    #         reconstruction_test.append(reconstruction[i].cpu().detach().numpy())
    #         mu_test.append(mu[i].cpu().detach().numpy())
    #         z_test.append(z[i].cpu().detach().numpy())
    #     bce_loss = criterion(reconstruction, data_test[:,1])
    #     loss = final_loss(bce_loss, mu, logvar,beta_used)
    #     loss_test+=loss.item()
    # loss_test=loss_test/len_test    

    print('Epoch:'+str(epoch)+', Training: '+str(round(loss_train,10)))
    # print('Epoch:'+str(epoch)+', Testing: '+str(round(loss_test,10)))
    print('Epoch:'+str(epoch)+', Timing: '+str(round(t1-t0,1)))
    print('current learning rate:',learning_rate)
    print('current beta:',beta_used)
    clear_output(wait=True)
    # L_test.append(loss_test)
    L_train.append(loss_train)

    if (epoch) % epoch_milestone==0:
        model.eval()
        print('saving model...')
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),},
                   method_export+'model_'+Version+'_epoch'+str(epoch)+'.pth')






