# prediction
import torch 
import matplotlib.pyplot as plt
from model import Net2

model = Net2()
model.load_state_dict(torch.load('kvasir.pt'))
import numpy as np


import math
#myImageDataset2 = ImageDataset_segment_test('/content/drive/MyDrive/TEST/')
model.eval()   # Set model to evaluate mode

# # sửa đây nhé: ony "test_dataset{i}", i =1,2,3,4,5
# test_loader = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=0)
####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_loader = None
#fg , ax = plt.subplots(len(test_loader), 3 )
#fg.suptitle("Compare result training set")
        
for i, (orgs , inputs, labels) in enumerate(test_loader):
  plt.rcParams['figure.figsize'] = [10, 8]
  
  inputs = inputs.permute(0,3,1,2).float().to(device)
  
  pred = model(inputs)
  pred = torch.sigmoid(pred).squeeze(0).data.cpu().numpy()
  pred_ = pred > 0.5
  pred_ = pred_.astype(float)

  # ax[i,0].imshow(inputs.squeeze(0).permute(1,2,0).data.cpu().numpy())
  # ax[i,0].set_xlabel("Original image {}")
    
  # ax[i,1].imshow(np.concatenate([pred_, pred_, pred_], axis = 0).transpose(1,2,0))
  # ax[i,1].set_xlabel("Predict")

  labels_ = labels.float().squeeze(0)
  #labels_ = torch.cat([labels_,labels_,labels_], axis = 2)
  labels_ = labels_.data.cpu().numpy()
  # ax[i,2].imshow(np.concatenate([labels_,labels_,labels_], axis = 2 ))
  # ax[i,2].set_xlabel("Ground Truth")
  plt.figure(i)
  plt.clf
  plt.subplot(1,3,1).imshow((orgs).squeeze(0).data.cpu().numpy()),plt.title('Image')
  # plt.subplot(1,3,2).imshow(np.concatenate([labels_,labels_,labels_], axis = 2 )),plt.title('Mask')
  plt.subplot(1,3,2).imshow(labels_),plt.title('Mask')
  plt.subplot(1,3,3).imshow(np.concatenate([pred_, pred_, pred_], axis = 0).transpose(1,2,0)),plt.title('pred.')
