from model import *
from dataset import Load_Data, train_transform, val_transform
from metrics import dice_coef, iou_score, calc_loss, dice_loss
import torch
import copy
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def algin_mask(mask):
  cmap = np.all(np.equal(mask,[255,255,255]), axis = -1)
  return cmap

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        if k =='iou' or k == 'dice_coef':
          outputs.append("{}: {:4f}".format(k, metrics[k]))
        else:
          outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))


    print("{}: {}".format(phase, ", ".join(outputs)))

# val is on TEST folder
folder_train  = '/content/drive/MyDrive/Kvasir-new/TrainDataset' 
folder_val  = '/content/drive/MyDrive/Kvasir-new/ValDataset' 
folder_test_1  = '/content/drive/MyDrive/Kvasir-new/TestDataset/Kvasir' 
folder_test_2  = '/content/drive/MyDrive/Kvasir-new/TestDataset/ETIS-LaribPolypDB' 
folder_test_3  = '/content/drive/MyDrive/Kvasir-new/TestDataset/CVC-ColonDB' 
folder_test_4  = '/content/drive/MyDrive/Kvasir-new/TestDataset/CVC-ClinicDB' 
folder_test_5  = '/content/drive/MyDrive/Kvasir-new/TestDataset/CVC-300' 

train_dataset = Load_Data(folder_train, transform = train_transform) 
val_dataset = Load_Data(folder_val, transform = val_transform) 
test_dataset1 = Load_Data(folder_test_1, transform = val_transform)
test_dataset2 = Load_Data(folder_test_2, transform = val_transform)
test_dataset3 = Load_Data(folder_test_3, transform = val_transform)
test_dataset4 = Load_Data(folder_test_4, transform = val_transform)
test_dataset5 = Load_Data(folder_test_5, transform = val_transform)
#train_2 = Load_Data(folder_path, transform = val_transform)
batch_size = 4

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0),
    'test_1': DataLoader(test_dataset1, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0),
    'test_2': DataLoader(test_dataset2, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0),
    'test_3': DataLoader(test_dataset3, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0),
    'test_4': DataLoader(test_dataset4, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0),
    'test_5': DataLoader(test_dataset5, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=0)

}

dataset_sizes = {
    x: len(dataloaders[x]) for x in dataloaders.keys()
}


dataset_sizes
def train_model(model, optimizer, scheduler, history, patience, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    counter = 0
    best_dice = [0,0,0,0,0]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test_1', 'test_2', 'test_3', 'test_4', 'test_5']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            dice = 0
            iou = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                labels = np.asarray(labels)
                labels = algin_mask(labels)
                labels = torch.tensor(labels)
                inputs = inputs.permute(0, 3, 1, 2).float().to(device)
                labels = labels.unsqueeze(1).float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    dice += dice_coef(outputs, labels)
                    iou += iou_score(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            metrics['iou'] = iou/len(dataloaders[phase])
            metrics['dice_coef'] = dice/len(dataloaders[phase])
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            history['dice_coef'][phase].append(dice/len(dataloaders[phase]))
            history['iou'][phase].append(iou/len(dataloaders[phase]))
            history['loss'][phase].append(epoch_loss)

            if phase == 'val':

                if epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                  counter = counter + 1
            if phase == 'test_1':
                if dice/len(dataloaders[phase]) > best_dice[0]:
                    best_dice[0] = dice/len(dataloaders[phase])
                    print('save best weight of kvasir')
                    torch.save(model.state_dict(),'kvasir.pt')
            if phase == 'test_2':
                if dice/len(dataloaders[phase]) > best_dice[1]:
                    best_dice[1] = dice/len(dataloaders[phase])
                    print('save best weight of etis')
                    torch.save(model.state_dict(),'etis.pt')
            if phase == 'test_3':
                if dice/len(dataloaders[phase]) > best_dice[2]:
                    best_dice[2] = dice/len(dataloaders[phase])
                    print('save best weight of clonDB')
                    torch.save(model.state_dict(),'ClonDB.pt')
            if phase == 'test_4':
                if dice/len(dataloaders[phase]) > best_dice[3]:
                    best_dice[3] = dice/len(dataloaders[phase])
                    print('save best weight of clinDB')
                    torch.save(model.state_dict(),'ClinDB.pt')
            if phase == 'test_5':
                if dice/len(dataloaders[phase]) > best_dice[4]:
                    best_dice[4] = dice/len(dataloaders[phase])
                    print('save best weight of CVC-300')
                    torch.save(model.state_dict(),'CVC300.pt')                    
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if counter > patience:
                  break
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

