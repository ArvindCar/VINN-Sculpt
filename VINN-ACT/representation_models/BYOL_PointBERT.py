'''
modified Phil Wang's code
url: https://github.com/lucidrains/byol-pytorch
'''
import torch
from torch import nn
from torchvision import models
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np

import sys
import wandb
import random
import argparse
# from byol_pytorch import BYOL


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--root_dir', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--img_size', type=int)
parser.add_argument('--hidden_layer', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--wandb', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--pretrained', type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--extension', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--bc_model', type=str)

class CustomPointCloudAugmentation:
    def __init__(self, pointcloud_size=2048):
        self.pointcloud_size = pointcloud_size

    def __call__(self, pointcloud):
        # Perform custom augmentation on the point cloud

        # # Example: Randomly shuffle the order of points
        # pointcloud = self.shuffle_points(pointcloud)

        # # Example: Randomly drop points to achieve the desired pointcloud_size
        # pointcloud = self.random_dropout(pointcloud)

        # Additional custom augmentations can be added here
        pointcloud = self.shift_points(pointcloud)

        return pointcloud

    # def shuffle_points(self, pointcloud):
    #     # Randomly shuffle the order of points in the point cloud
    #     indices = torch.randperm(pointcloud.size(0))
    #     return pointcloud[indices]

    # def random_dropout(self, pointcloud):
    #     # Randomly drop points to achieve the desired pointcloud_size
    #     if pointcloud.size(0) > self.pointcloud_size:
    #         indices_to_keep = torch.randperm(pointcloud.size(0))[:self.pointcloud_size]
    #         pointcloud = pointcloud[indices_to_keep]
    #     return pointcloud

    def shift_points(self, pointcloud, max_shift=0.01):
        # Slightly shift each point in the point cloud
        shift_amounts = torch.rand_like(pointcloud) * 2 * max_shift - max_shift
        pointcloud = pointcloud + shift_amounts
        return pointcloud



if __name__ == '__main__':
    args = parser.parse_args()
    params = vars(args)
    params['representation'] = 1
    
    wandb.init(project = 'VINN-Sculpt', entity='arvindcar')
    if(params['wandb'] == 1):
        wandb.init(project = 'PointBERT_v2_' + params['extension'], entity="cmu_MAIL")
        wandb.run.name = "Batch_Size_" + str(params['batch_size'])

    sys.path.append(params['root_dir'] + 'dataloaders')
    from XDataset_PC import XDataset_PC

    pointbert_path = '/'.join(params['root_dir'].split('/')[:-4]) + '/point_cloud_embedding'
    sys.path.append(pointbert_path)
    from PointBERTwEncoder import PointBERTWithProjection

    sys.path.append(params['root_dir'] + 'representation_models')
    from byol_pytorch_pointbert import BYOL
    


    PCAug = CustomPointCloudAugmentation()
    customAug = T.Compose([T.Lambda(lambda x: PCAug(x))])

    if(params['dataset'] == 'X_Datasets'):
        img_data = XDataset_PC(params, None)
    # if(params['dataset'] == 'PushDataset' or params['dataset'] == 'StackDataset'):
    #     img_data = PushDataset(params, None)

    # if(params['pretrained'] == 1):
    #     model = models.resnet50(pretrained=True)
    # else:
    #     model = models.resnet50(pretrained=False)
    model = PointBERTWithProjection(pointbert_path)

    if(params['gpu'] == 1):
        device = torch.device('cuda')
        model = model.to(device)
        dataloader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers = 8)
    else:
        dataloader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True)

    learner = BYOL(
        model,
        image_size = params['img_size'],
        hidden_layer = params['hidden_layer'],
        augment_fn = customAug
    )

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)


    epochs = params['epochs']
    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(dataloader, 0):
            if(params['gpu'] == 1):
                loss = learner(data.float().to(device))
            else:
                loss = learner(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            epoch_loss += loss.item()*data.shape[0]

        print(epoch_loss / len(img_data))
        if(params['wandb'] == 1):
            wandb.log({'train loss': epoch_loss / len(img_data)})

        if(epoch % 20  == 0):
            torch.save({'model_state_dict': model.state_dict()
                    }, params['save_dir']+'BYOL_PC_'+str(epoch)+'_'+params['extension']+'_batch_'+str(params['batch_size'])+'.pt')