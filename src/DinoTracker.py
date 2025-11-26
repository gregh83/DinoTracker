import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QGridLayout, QPushButton, QFileDialog, QComboBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph as pg
from PIL import Image, ImageOps
from io import BytesIO
import os
from PIL import Image
from io import BytesIO
from PyQt6.QtWidgets import QDial

DinoTrackerVersion='1.0'

try:
    
    from sklearn.manifold import TSNE
except ImportError:
    #sklearn is not natively available on arm Windows, so this is a fallback, if import fails
    class TSNE:
        def __init__(self, n_components=2, **kwargs):
            self.n_components = n_components
            print('t-SNE import failed -> PCA fallback')
        def fit_transform(self, X):
            X = X - np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            components = eigvecs[:, :self.n_components]
            return np.dot(X, components)


from PyQt6.QtWidgets import QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem
)
from PyQt6.QtGui import QPixmap, QPen, QPolygonF
from PyQt6.QtCore import Qt, QPointF
import sys
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtWidgets import QGraphicsPathItem
from PyQt6.QtGui import QPainterPath
from PyQt6.QtWidgets import QFileDialog



features=50
filter0,filter1,filter2=60,60,60
dfilter0,dfilter1,dfilter2=40,25,10
conv_size=3
pad=1
stridepool_size=2
pool_size=2
IMG_SIZE=100

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
        self.map_decode=nn.Linear(in_features=features, out_features=50*50)#100)
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

feature_names = ['1. Ground contact','2. Digit spread','3. Digit attachment','4. Heel load',
                 '5. Digit/heel emphasis','6. Loading position',
                 '7. Heel position','8. L/R-load','9. Left or right','10. Rotation']

active_idxs=[19,4,3,20,22,13,47,33,32,48]

device = torch.device("cpu")
latent_vector = torch.zeros(50,dtype=torch.float32)

colors = [
    (255, 179, 186),
    (255, 223, 186),
    (255, 255, 186),
    (186, 255, 201),
    (186, 225, 255),
    (201, 186, 255),
    (255, 186, 255)
]

ornithopod_end=662
theropod_end=1640
bird_end=1855
stompy_end=1949

def compute_class_means(L_features):
    mean_o = np.mean(L_features[0:ornithopod_end], axis=0)
    mean_t = np.mean(L_features[ornithopod_end:theropod_end], axis=0)
    mean_b = np.mean(L_features[theropod_end:bird_end], axis=0)
    mean_s = np.mean(L_features[bird_end:stompy_end], axis=0)
    return mean_o, mean_t, mean_b, mean_s

def load_data():
    images = np.load('./data/images_compressed.npz')['images']
    L_features = np.load('./models/mu.npy')
    names = np.load('./data/names.npy')
    return images, L_features, names

images, L_features, names = load_data()
mean_o, mean_t, mean_b, mean_s = compute_class_means(L_features)


model = mixerVAE()
checkpoint = torch.load('./models/model_BETA15_BIG_3k_shuffle_epoch1000.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

blank_array = np.zeros((100, 100), dtype=np.uint8)



def get_black_pixmap():
    blank_array = np.zeros((100, 100), dtype=np.uint8)
    pil_img = Image.fromarray(blank_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    qimg = QImage.fromData(buffer.getvalue())
    return QPixmap.fromImage(qimg)

def auto_invert_if_needed_np(image_array):
    corners = [
        image_array[0, 0],
        image_array[0, -1],
        image_array[-1, 0],
        image_array[-1, -1]
    ]
    if sum(corners) >= 3:
        image_array = 1.0 - image_array
    return image_array


    
class VAEApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DinoTracker v"+DinoTrackerVersion)
        pixmap_logo = QPixmap("./data/Tone_logo_small.png") 
        logo_label = QLabel()
        logo_label.setFixedSize(100, 100)

        scaled_pixmap = pixmap_logo.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)

        version_label = QLabel("DinoTracker v"+DinoTrackerVersion)
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        version_label.setStyleSheet("font-size: 14px;")

        bottom_right_layout = QHBoxLayout()
        bottom_right_layout.addWidget(version_label)
        bottom_right_layout.addWidget(logo_label)
        bottom_right_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)


        self.tsne_button = QPushButton("Run t-SNE")
        self.tsne_button.clicked.connect(self.run_tsne_and_show_plot)

        self.draw_button = QPushButton("Draw silhouette")
        self.draw_button.clicked.connect(self.open_drawing_tool)
        
        self.neighbor_layout = QVBoxLayout()
        
        self.neighbor_label = QLabel("Seven closest Neighbors")
        self.neighbor_label.setStyleSheet("font-size: 16px;")
        self.neighbor_label.setFixedSize(250, 20)
        self.neighbor_layout.addWidget(self.neighbor_label)
        
        self.neighbor_widgets = []  

        f_size=200
        self.original_image_widget = QLabel()
        self.original_image_widget.setFixedSize(f_size, f_size)
        self.original_image_widget.setScaledContents(True)
        
        self.reconstructed_image_widget = QLabel()
        self.reconstructed_image_widget.setFixedSize(f_size,f_size)
        self.reconstructed_image_widget.setScaledContents(True)
        
        self.original_label = QLabel("Original")
        self.original_label.setStyleSheet("font-size: 16px;")
        self.original_label.setFixedSize(120, 20)
        self.reconstructed_label = QLabel("Reconstruction")
        self.reconstructed_label.setStyleSheet("font-size: 16px;")
        self.reconstructed_label.setFixedSize(120, 20)
        
        reconstructed_layout = QVBoxLayout()
        reconstructed_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        reconstructed_layout.addWidget(self.reconstructed_label)
        reconstructed_layout.addWidget(self.reconstructed_image_widget)
        

        self.original_images = {} 
        self.latent_vectors = {}
        self.image_names = []

        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self.load_selected_encoding)

        self.import_button = QPushButton("Import PNG Images")
        self.import_button.clicked.connect(self.import_images)
        
        self.sliders = []
        self.latent_z = latent_vector.clone()
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid_label=QLabel("Features")
        grid_label.setStyleSheet("font-size: 16px;")
        grid.addWidget(grid_label)
        for i, idx in enumerate(active_idxs):
            label = QLabel(feature_names[i])
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 12px;
                    background: #d0d0d0;
                    border-radius: 6px;
                }
            
                QSlider::handle:horizontal {
                    background: #0078d7;
                    border: 1px solid #5c5c5c;
                    width: 30px;
                    height: 30px;
                    margin: -10px 0;
                    border-radius: 15px;
                }
            """)
            slider.setMinimum(-30)
            slider.setMaximum(30)
            slider.setValue(0)
            slider.setSingleStep(1)
            slider.valueChanged.connect(self.update_reconstruction)
            slider.setFixedSize(300, 50)
            self.sliders.append((slider, idx))

            grid.addWidget(label, i+1, 0)
            grid.addWidget(slider, i+1, 1)


        original_layout = QVBoxLayout()
        original_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        original_layout.addLayout(reconstructed_layout)
        
        original_layout.addWidget(self.original_label)
        original_layout.addWidget(self.original_image_widget)
        
        original_layout.addWidget(self.import_button)
        original_layout.addWidget(self.selector)
        original_layout.addWidget(self.tsne_button)
        original_layout.addWidget(self.draw_button)
        

        image_layout = QHBoxLayout()
        image_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        image_layout.addLayout(original_layout)

        image_layout.addLayout(self.neighbor_layout)

        feature_space_layout=QVBoxLayout()

        self.feature_label = QLabel("Feature Space")
        self.feature_label.setStyleSheet("font-size: 16px;")
        self.feature_label.setFixedSize(140, 20)

        
        feature_space_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        feature_space_layout.addWidget(self.feature_label)
        self.image_label = QLabel()
       

        self.original_image_widget.setPixmap(get_black_pixmap())
        self.show_neighbors(self.latent_z.clone())
        
        features_neigbors,names_neigbors=self.get_neighbor_features(self.latent_z.clone())
        plot_widget = pg.PlotWidget()
        plot_widget.setFixedSize(400, 250)
    
        self.legend = pg.LegendItem(offset=(-1, 10)) 
        self.legend.setParentItem(plot_widget.plotItem)
        curve=plot_widget.plot(range(1,20), [0 for i in range(1,20)],pen='black')
         
        self.curves=[]
        self.curves.append(curve)
        
        curve=plot_widget.plot(range(1,11), self.latent_z.clone()[active_idxs],pen=pg.mkPen(color='magenta', width=10))
        self.curve_group=plot_widget.plot(range(1,11), self.latent_z.clone()[active_idxs],pen=pg.mkPen(color='gray', width=5,style=Qt.PenStyle.DashLine))
        self.curves.append(curve)
        
        for idx in range(7):
            curve=plot_widget.plot(range(1,11),features_neigbors[idx],pen=pg.mkPen(colors[idx], width=2))
            self.curves.append(curve)
            self.legend.addItem(curve, names_neigbors[idx][-30:-4])
        feature_space_layout.addWidget(plot_widget)
        

        plot_widget2 = pg.PlotWidget()
        plot_widget2.setFixedSize(400, 250)
        self.legend2 = pg.LegendItem(offset=(-1, 10))
        self.legend2.setParentItem(plot_widget2.plotItem)
        curve2=plot_widget2.plot(range(1,20), [0 for i in range(1,20)],pen='black')
        self.curves2=[]
        curve2=plot_widget2.plot(range(1,11), self.latent_z.clone()[active_idxs],pen=pg.mkPen(color='magenta', width=10))
        self.curves2.append(curve2)
        self.curve2_group=plot_widget2.plot(range(1,11), self.latent_z.clone()[active_idxs],pen=pg.mkPen(color='gray', width=5,style=Qt.PenStyle.DashLine))
        self.legend2.addItem(self.curve2_group, 'Imported Group Avg')
        
        
        curve2=plot_widget2.plot(range(1,11),mean_o,pen=pg.mkPen(color='green', width=2))
        self.curves2.append(curve2)
        cur_distance_sample=np.linalg.norm(mean_o-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(curve2, 'Ornithopod:'+str(round(cur_distance_sample,2)))

        curve2=plot_widget2.plot(range(1,11),mean_t,pen=pg.mkPen(color='red', width=2))
        self.curves2.append(curve2)
        cur_distance_sample=np.linalg.norm(mean_t-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(curve2, 'Theropod:'+str(round(cur_distance_sample,2)))

        curve2=plot_widget2.plot(range(1,11),mean_b,pen=pg.mkPen(color='blue', width=2))
        self.curves2.append(curve2)
        cur_distance_sample=np.linalg.norm(mean_b-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(curve2, 'Bird:'+str(round(cur_distance_sample,2)))

        
        curve2=plot_widget2.plot(range(1,11),mean_s,pen=pg.mkPen(color='orange', width=2))
        self.curves2.append(curve2)
        cur_distance_sample=np.linalg.norm(mean_s-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(curve2, 'Tetrapod:'+str(round(cur_distance_sample,2)))
        
        feature_space_layout.addWidget(plot_widget2)
        
        
        self.update_reconstruction()
        

        main_layout = QHBoxLayout()
        main_layout.addLayout(grid)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.image_label)

        feature_space_layout.addLayout(bottom_right_layout)


        
        main_layout.addLayout(feature_space_layout)

        
        self.setLayout(main_layout)


    def run_tsne_and_show_plot(self):
        latent_vectors = self.get_latent_vectors() 
        if not np.isfinite(latent_vectors).all():
            print("Warning: Latent vectors contain NaN or inf values!")
        tsne = TSNE(n_components=2,learning_rate='auto',init='random',perplexity=26,verbose=1,
            max_iter=100000,n_iter_without_progress=300,n_jobs=-1,early_exaggeration=12,random_state=0)
        tsne_result = tsne.fit_transform(latent_vectors)
        self.show_tsne_window(tsne_result)

    def show_tsne_window(self, tsne_result):
        self.tsne_window = TSNEWindow(tsne_result)
        self.tsne_window.show()

    def open_drawing_tool(self):
        self.drawing_window = launch_drawing_tool(self)

    def get_latent_vectors(self):
        print(L_features.shape)
        cur_group=np.array([tensor[active_idxs].detach().cpu().numpy() for tensor in self.latent_vectors.values()])
        print(cur_group.shape)


        cur_z=np.array([self.latent_z.clone()[active_idxs].detach().cpu().numpy()])
        print(cur_z.shape)
        out=np.append(L_features,cur_z,axis=0)
        
        if len(cur_group)>0:
            out=np.append(out,cur_group,axis=0)
        print(out.shape)
        out=out[:, :8]
        print(out.shape)
        return out


    
    def update_feature_maps(self):
        self.legend.clear()
        features_neigbors,names_neigbors=self.get_neighbor_features(self.latent_z.clone())
        self.curves[1].setData(range(1,11),self.latent_z.clone()[active_idxs])
        self.curves2[0].setData(range(1,11),self.latent_z.clone()[active_idxs])
        for idx in range(2,9):
            self.curves[idx].setData(range(1,11),features_neigbors[idx-2])
            self.legend.addItem(self.curves[idx], names_neigbors[idx-2][-25:-4])
        
        self.legend2.clear()

        self.legend2.addItem(self.curve2_group, 'Imported Group Avg')
        
        cur_distance_sample=np.linalg.norm(mean_o-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(self.curves2[1], 'Ornithopod:'+str(round(cur_distance_sample,2)))

        cur_distance_sample=np.linalg.norm(mean_t-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(self.curves2[2], 'Theropod:'+str(round(cur_distance_sample,2)))

        cur_distance_sample=np.linalg.norm(mean_b-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(self.curves2[3], 'Bird:'+str(round(cur_distance_sample,2)))

        cur_distance_sample=np.linalg.norm(mean_s-self.latent_z.clone()[active_idxs].numpy())
        self.legend2.addItem(self.curves2[4], 'Tetrapod:'+str(round(cur_distance_sample,2)))
        
        if len(self.latent_vectors)>1:
            avg=torch.stack(list(self.latent_vectors.values())).mean(dim=0)[active_idxs]
            self.curve_group.setData(range(1,11),avg)
            self.curve2_group.setData(range(1,11),avg)

                
    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select PNG Images", "", "Images (*.png)")
        for file_path in files:
            name = os.path.basename(file_path)
            image = Image.open(file_path).convert("L")
            image = image.resize((100, 100), Image.NEAREST)
            image = ImageOps.invert(image)
            np_array = np.array(image) / 255.0
            np_array = auto_invert_if_needed_np(np_array)
            image_tensor = torch.tensor(np_array, dtype=torch.float32).to(device)
            self.original_images[name] = image_tensor
            with torch.no_grad():
                z_from_image = model(image_tensor.unsqueeze(0), 'x')[1][0]
                self.latent_vectors[name] = z_from_image
                if name not in self.image_names:
                    self.image_names.append(name)
                    self.selector.addItem(name)
        self.selector.setCurrentIndex(len(self.image_names) - 1)

    def get_closest_neighbors(self, query_z, top_k=7):
        query_sub = query_z[active_idxs[:8]]
        distances = []
        for i, ref_z in enumerate(L_features):
            ref_sub = torch.tensor(ref_z)[:8]
            dist = torch.norm(query_sub - ref_sub)
            distances.append((dist.item(), i))
    
        return sorted(distances, key=lambda x: x[0])[:top_k]

    def get_neighbor_features(self, query_z, top_k=7):
        closest = self.get_closest_neighbors(query_z, top_k)
        neighbor_indices = [idx for _, idx in closest]
        neighbor_features = [L_features[idx] for idx in neighbor_indices]
        neighbor_names = [names[idx] for idx in neighbor_indices]
        return neighbor_features, neighbor_names
    
    
    
    def show_neighbors(self, query_z):
        for widget in self.neighbor_widgets:
            self.neighbor_layout.removeWidget(widget)
            widget.deleteLater()
        self.neighbor_widgets = []
    
        neighbors = self.get_closest_neighbors(query_z)
    
        for dist, idx in neighbors:
            
            img_tensor = images[idx]
            pil_img = Image.fromarray((img_tensor * 255).astype(np.uint8))
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            qimg = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimg)

            container = QWidget()
            h_layout = QHBoxLayout(container)
            
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setScaledContents(True)
            image_label.setFixedSize(70, 70)

            track_label='None'
            if idx<ornithopod_end:
                track_label="ornithopod"
            elif idx<theropod_end:
                track_label="theropod"
            elif idx<bird_end:
                track_label="bird"
            elif idx<stompy_end:   
                track_label="tetrapod"
            
            text_label = QLabel(f"{names[idx][-20:-4]}\nDist: {round(dist, 2)}\n{track_label}")
            text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            
            h_layout.addWidget(image_label)
            h_layout.addWidget(text_label)
            
            self.neighbor_layout.addWidget(container)
            self.neighbor_widgets.append(container)

            
    def load_selected_encoding(self):
        name = self.selector.currentText()
        if name in self.latent_vectors:
            cur_z=self.latent_vectors[name].clone()
            for i, idx in enumerate(active_idxs):
                self.sliders[i][0].setValue(int(cur_z[idx].item()*10))
            
            self.update_reconstruction()

    def update_reconstruction(self):
        for slider, idx in self.sliders:
            self.latent_z[idx] = slider.value() / 10.0
    
        with torch.no_grad():
            generated = model(self.latent_z.unsqueeze(0), 'z')
            img = (generated.squeeze() >= 0.5).float().numpy()
    
        recon_img = Image.fromarray((img * 255).astype(np.uint8))
        buffer = BytesIO()
        recon_img.save(buffer, format="PNG")
        qimg = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        self.reconstructed_image_widget.setPixmap(pixmap)
    
        name = self.selector.currentText()
        if name in self.original_images:
            orig_tensor = self.original_images[name].numpy()
            orig_img = Image.fromarray((orig_tensor * 255).astype(np.uint8))
            buffer = BytesIO()
            orig_img.save(buffer, format="PNG")
            qimg_orig = QImage.fromData(buffer.getvalue())
            pixmap_orig = QPixmap.fromImage(qimg_orig)
            self.original_image_widget.setPixmap(pixmap_orig)
        
        self.show_neighbors(self.latent_z.clone())
        self.update_feature_maps()
        
class TSNEWindow(QMainWindow):
    def __init__(self, tsne_result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("t-SNE Visualization")

        fig = Figure(figsize=(12, 12))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.scatter(tsne_result[:, 0][0:ornithopod_end], tsne_result[:, 1][0:ornithopod_end], alpha=0.5,color='limegreen',label='Ornithopod')
        ax.scatter(tsne_result[:, 0][ornithopod_end:theropod_end], tsne_result[:, 1][ornithopod_end:theropod_end], alpha=0.5,color='red',label='Theropod')
        ax.scatter(tsne_result[:, 0][theropod_end:bird_end], tsne_result[:, 1][theropod_end:bird_end], alpha=0.5,color='blue',label='Bird')
        ax.scatter(tsne_result[:, 0][bird_end:stompy_end], tsne_result[:, 1][bird_end:stompy_end], alpha=0.5,color='purple',label='Tetrapod')
    
        ax.scatter(tsne_result[:, 0][stompy_end:stompy_end+1], tsne_result[:, 1][stompy_end:stompy_end+1], alpha=1,color='magenta',marker='s',s=100,label='Current setting')
        ax.scatter(tsne_result[:, 0][stompy_end+1:], tsne_result[:, 1][stompy_end+1:], alpha=1,color='black',marker='s',s=100,label='Imported images')
        ax.legend()
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.grid(linestyle='--',alpha=0.5)
        

        self.setCentralWidget(canvas)           


def crop_and_pad_silhouette(arr, frame_ratio=0.05):
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    height = rmax - rmin + 1
    width = cmax - cmin + 1
    size = max(height, width)

    r_center = (rmin + rmax) // 2
    c_center = (cmin + cmax) // 2
    half_size = size // 2

    r_start = max(0, r_center - half_size)
    r_end = min(arr.shape[0], r_start + size)
    r_start = r_end - size 

    c_start = max(0, c_center - half_size)
    c_end = min(arr.shape[1], c_start + size)
    c_start = c_end - size 

    cropped = arr[r_start:r_end, c_start:c_end]

    pad = int(size * frame_ratio)
    padded = np.pad(cropped, pad_width=pad, mode='constant', constant_values=0)

    return padded


class LassoGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.scene = scene
        self.drawing = False
        self.current_points = []
        self.pen = QPen(Qt.GlobalColor.red, 2)
        self.temp_path_item = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            scene_pos = self.mapToScene(event.pos())
            self.current_points = [scene_pos]
            path = QPainterPath(scene_pos)
            self.temp_path_item = QGraphicsPathItem(path)
            self.temp_path_item.setPen(self.pen)
            self.scene.addItem(self.temp_path_item)

    def mouseMoveEvent(self, event):
        if self.drawing:
            scene_pos = self.mapToScene(event.pos())
            self.current_points.append(scene_pos)
            path = QPainterPath(self.current_points[0])
            for point in self.current_points[1:]:
                path.lineTo(point)
            self.temp_path_item.setPath(path)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            if self.temp_path_item:
                self.scene.removeItem(self.temp_path_item)
                self.temp_path_item = None
            if len(self.current_points) > 2:
                polygon = QPolygonF(self.current_points + [self.current_points[0]])
                item = QGraphicsPolygonItem(polygon)
                item.setPen(self.pen)
                self.scene.addItem(item)
            self.current_points = []



def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)
    width = qimage.width()
    height = qimage.height()

    ptr = qimage.bits()
    ptr.setsize(height * width)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
    binary_arr = (arr > 127).astype(np.uint8)
    return binary_arr


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    arr = (arr * 255).astype(np.uint8)
    height, width = arr.shape
    qimage = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8)
    return qimage.copy() 


class LassoView(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.scene = QGraphicsScene()
        self.view = LassoGraphicsView(self.scene)
        self.view.scale(1.5, 1.5)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.button_silhouette = QPushButton("Create Silhouette")
        self.button_silhouette.clicked.connect(self.create_silhouette)
        self.button_clear = QPushButton("Clear All")
        self.button_clear.clicked.connect(self.clear_all)
        layout.addWidget(self.button_silhouette)
        layout.addWidget(self.button_clear)
        self.setLayout(layout)
        original_pixmap = QPixmap(image_path)
        width = original_pixmap.width()
        height = original_pixmap.height()
        side = min(width, height)
        x_offset = (width - side) // 2
        y_offset = (height - side) // 2
        square_pixmap = original_pixmap.copy(x_offset, y_offset, side, side)
        resized_pixmap = square_pixmap.scaled(400, 400, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.pixmap = resized_pixmap
        
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)
    def clear_all(self):
        for item in self.scene.items():
            if isinstance(item, QGraphicsPolygonItem):
                self.scene.removeItem(item)
        if hasattr(self.view, "temp_path_item") and self.view.temp_path_item:
            self.scene.removeItem(self.view.temp_path_item)
            self.view.temp_path_item = None
 
    def create_silhouette(self):
        width = self.pixmap.width()
        height = self.pixmap.height()
        mask_image = QImage(width, height, QImage.Format.Format_Grayscale8)
        mask_image.fill(0)  # All black
    
        painter = QPainter(mask_image)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
    
        for item in self.scene.items():
            if isinstance(item, QGraphicsPolygonItem):
                painter.drawPolygon(item.polygon())
        painter.end()
    
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Silhouette",
            "",
            "PNG Files (*.png)"
        )
    
        if not file_path:
            print("Save cancelled.")
            return  
        mask_image_old=mask_image.copy()
        print(mask_image)
        numpy_mask=qimage_to_numpy(mask_image)
        print(numpy_mask.shape)
        cp_numpy_mask=crop_and_pad_silhouette(numpy_mask)
        print(cp_numpy_mask.shape)
        mask_image=numpy_to_qimage(cp_numpy_mask)
        mask_image.save(file_path)
        print(f"Silhouette saved as '{file_path}'")

        self.scene.removeItem(self.image_item)
        silhouette_pixmap = QPixmap.fromImage(mask_image_old)
        self.image_item = QGraphicsPixmapItem(silhouette_pixmap)
        self.scene.addItem(self.image_item)


from PyQt6.QtWidgets import QFileDialog

def launch_drawing_tool(parent=None):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select Dinosaur Footprint Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    )

    if file_path:
        view = LassoView(file_path)
        view.setWindowTitle("Lasso Drawing Tool")
        view.show()
        return view 
    else:
        print("No image selected.")
        return None


app = QApplication(sys.argv)
window = VAEApp()
window.show()
sys.exit(app.exec())


