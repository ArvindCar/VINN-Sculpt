import torch
import sys
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import robomail.vision as vis
import open3d as o3d
from PIL import Image
import torchvision.transforms as T

class VINN_Img():
    
    def dist_metric(self, x,y):
        return(torch.norm(x-y).item())

    def calculate_action(self, dist_list,k):
        translation = torch.tensor([0.0,0.0,0.0])
        rotation = torch.tensor([0.0])
        gripper = torch.tensor([0.0])
        top_k_weights = torch.zeros((k,))
        for i in range(k):
            top_k_weights[i] = dist_list[i][0]

        top_k_weights = self.softmax(-1*top_k_weights)
        for i in range(k):
            translation = torch.add(top_k_weights[i]*dist_list[i][1], translation)
            rotation = torch.add(top_k_weights[i]*dist_list[i][2], rotation)
            gripper = torch.add(top_k_weights[i]*dist_list[i][3], gripper)
        action = torch.cat((translation, rotation, gripper))
        return(action)

    def extract_image(self, full_path):
        parts = full_path.split('/')
        return '/'.join(parts[-2:])
    
    def calculate_nearest_neighbors(self, img_embedding, dataset, k):
        loss = [0 for i in range(k)]
        selected_paths = []
        dist_list = []
        # print(len(dataset))
        for dataset_index in range(len(dataset)):

            dataset_embedding, dataset_translation, dataset_rotation, dataset_gripper, dataset_path = dataset[dataset_index]
            print("Translation:",dataset_translation)
            print("Rotation:",dataset_rotation)
            print("This:",dataset_gripper)
            distance = self.dist_metric(img_embedding, dataset_embedding)
            dist_list.append((distance, dataset_translation, dataset_rotation, dataset_gripper, dataset_path))

        dist_list = sorted(dist_list, key = lambda tup: tup[0])
        print("First Elem:",dist_list[0][1], dist_list[0][2], dist_list[0][3])
        pred_action = self.calculate_action(dist_list, k)
        selected_paths.append(([self.extract_image(dist_list[j][4]) for j in range(k)]))
        print("Nearest States:", selected_paths)
        print("Predicted Action:", pred_action)
        return pred_action


    def __init__(self, root_dir, chkpts):
        self.params = {}
        self.params['root_dir'] = root_dir  #'/home/arvindcar/MAIL_Lab/VINN/VINN-Sculpt/VINN-ACT/' # This was changed
        self.params['img_size'] = 2048
        self.params['layer'] = ''
        self.params['model'] = 'PointBERT'
        self.params['representation_model_path'] = chkpts # This was changed
        self.params['eval'] = 0
        self.params['representation'] = 0
        self.params['dataset'] = 'X_Datasets'
        self.params['architecture'] = ''
        self.params['t'] = 0


        sys.path.append(self.params['root_dir'] + 'representation_models')
        sys.path.append(self.params['root_dir'] + 'dataloaders')
        from run_model import Encoder
        from XDataset_PC import XDataset_PC

        self.encoder = Encoder(self.params)
        self.params['folder'] =  '/home/arvind/VINN-Sculpt/VINN-Sculpt/VINN-ACT/representation_data/X_all/train_all' # '/home/arvindcar/MAIL_Lab/VINN/VINN-Sculpt/VINN-ACT/representation_data/X_all/train_all'
        self.train_dataset = XDataset_PC(self.params, self.encoder)
        self.mseLoss = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=0)


    def next_action(self, pc2, pc3, pc4, pc5, goal, k=10, viz = False):
        pcl_vis = vis.Vision3D()
        img = pcl_vis.fuse_point_clouds(pc2, pc3, pc4, pc5, vis=False, no_transformation=False)
        # np.save(self.params['root_dir'] + '/' + 'CurrentState.npy', points)
        # img = np.load(self.params['root_dir'] + '/' + 'CurrentState.npy')
        
        if vis:
            pointcloud = pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(img)
            pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(img), 1)))
            pointcloud.points.extend(o3d.utility.Vector3dVector(goal))
            pointcloud.colors.extend(o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(goal), 1))))
            o3d.visualization.draw_geometries([pointcloud])

        img_tensor = torch.tensor(img) 
        goal_tensor = torch.tensor(goal)
        img_embedding = self.encoder.encode(img_tensor)[0]
        goal_embedding = self.encoder.encode(goal_tensor)[0]
        conditioned_embedding = torch.cat((img_embedding, goal_embedding), dim = 0)
        next_action = self.calculate_nearest_neighbors(conditioned_embedding, self.train_dataset, k)
        next_action = np.array(next_action)
        return next_action