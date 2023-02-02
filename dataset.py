import os
import random as rn
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

import cv2
from glob import glob

import matplotlib.pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)
val_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

from glob import glob
class Load_Data(Dataset): 
    def __init__(self, folder_path , transform = None,  *args,**kwargs): 
      super().__init__(*args,**kwargs) 
      self.fpaths = glob(folder_path+'/images/*.png') 
      self.transform = transform
    def __getitem__(self, idx):
      path = self.fpaths[idx]
      img = cv2.imread(path)
      img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)   
      mask= cv2.imread(path.replace('images', 'masks'))
      mask = cv2.cvtColor(mask , cv2.COLOR_BGR2RGB)   
      if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img_t = transformed["image"]
            mask = transformed["mask"]
      img_t = torch.tensor(img_t)
      mask = torch.tensor(mask)
      img = cv2.resize(img, (256,256)) 
      return [img,img_t,mask]
    def __len__(self): 
      return len(self.fpaths) 