import torch
import sys
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import robomail.vision as vis
import open3d as o3d

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
            gripper = torch.add(top_k_weights[i]*dist_list[i][3], translation)
        action = torch.cat((translation, rotation, gripper))
        return(action)

    def extract_image(self, full_path):
        parts = full_path.split('/')
        return '/'.join(parts[-2:])
    
    def calculate_nearest_neighbors(self, img_embedding, dataset, k):
        loss = [0 for i in range(k)]
        selected_paths = []
        for dataset_index in range(len(dataset)):

            dataset_embedding, dataset_translation, dataset_rotation, dataset_gripper, dataset_path = dataset[dataset_index]
            distance = self.dist_metric(img_embedding, dataset_embedding)
            dist_list.append((distance, dataset_translation, dataset_rotation, dataset_gripper, dataset_path))

        dist_list = sorted(dist_list, key = lambda tup: tup[0])
        pred_action = self.calculate_action(dist_list, k)
        selected_paths.append(([self.extract_image(dist_list[j][4]) for j in range(k)]))
        print(selected_paths)
        return pred_action


    def __init__(self, root_dir, chkpts_dir):
        params = {}
        params['root_dir'] = root_dir  #'/home/arvindcar/MAIL_Lab/VINN/VINN-Sculpt/VINN-ACT/' # This was changed
        params['img_size'] = 624
        params['layer'] = 'avgpool'
        params['model'] = 'BYOL'
        params['representation_model_path'] = chkpts_dir # This was changed
        params['eval'] = 0
        params['representation'] = 0
        params['dataset'] = 'X_Datasets'
        params['architecture'] = 'ResNet'
        params['t'] = 0


        sys.path.append(params['root_dir'] + 'representation_models')
        sys.path.append(params['root_dir'] + 'dataloaders')
        print(sys.path)
        from run_model import Encoder
        from XDataset import XDataset

        self.encoder = Encoder(params)
        params['folder'] =  '/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/VINN-ACT/representation_data/X_all/train_all' # '/home/arvindcar/MAIL_Lab/VINN/VINN-Sculpt/VINN-ACT/representation_data/X_all/train_all'
        self.train_dataset = XDataset(params, self.encoder)
        self.mseLoss = torch.nn.MSELoss()
        # ceLoss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=0)

    def next_action(self, pc2, pc3, pc4, pc5):
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window()
        pcl_vis = vis.Vision3D()
        pc2.transform(pcl_vis.camera_transforms[2])
        pc3.transform(pcl_vis.camera_transforms[3])
        pc4.transform(pcl_vis.camera_transforms[4])
        pc5.transform(pcl_vis.camera_transforms[5])

        # combine the point clouds
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = pc5.points
        pointcloud.colors = pc5.colors
        pointcloud.points.extend(pc2.points)
        pointcloud.colors.extend(pc2.colors)
        pointcloud.points.extend(pc3.points)
        pointcloud.colors.extend(pc3.colors)
        pointcloud.points.extend(pc4.points)
        pointcloud.colors.extend(pc4.colors)

        # remove statistical outliers
        pointcloud, ind = pointcloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )  
        
        # crop point cloud
        pointcloud = pcl_vis.remove_stage_grippers(pointcloud) # Change ind_z_upper in Vision3D.remove_stage_grippers if you want the stage to be in frame
        pointcloud = pcl_vis.remove_background(
        pointcloud, radius=0.15, center=np.array([0.6, -0.05, 0.3]) # change radius to 0.3 if you want the goal shape to be in frame
        )
        vis1.add_geometry(pointcloud)
        pc_img = vis1.capture_screen_float_buffer(True)
        vis1.remove_geometry(pointcloud)
        img_tensor = torch.tensor(pc_img)
        img_embedding = self.encoder.encode(img_tensor)
        next_action = self.calculate_nearest_neighbors(img_embedding, self.train_dataset, 50)
        next_action = np.array(next_action)
        return next_action


# import numpy as np
# import open3d as o3d
# import robomail.vision as vis
# from matplotlib import pyplot as plt
# import os
# import json

# from augmentation_utils import *

# vis1 = o3d.visualization.Visualizer()
# vis1.create_window()

# data_path = '/home/aesee/CMU/MAIL_Lab/Human Demos/Dec14_Human_Demos_Images/Dec14_Human_Demos_Raw/X/' # Directory where all the collected data is
# n_traj = 10
# n_rot_aug = 6

# for j in range(n_rot_aug):

#     for i in range(n_traj): # Change to number of trajectories
#         save_path = data_path + 'run_' + str(j*n_traj+i) + '/images' # Where you want this trajectory saved
#         save_path_actions = data_path + 'run_' + str(j*n_traj+i) + '/labels.json'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         traj_path = data_path + 'Trajectory' + str(i) + '/' 
#         states = [state for state in os.listdir(traj_path) if state.startswith("State")] 
#         actions_dict = {}

#         for s, state in enumerate(states): # Iterates through states within a given trajextory
#             state_path = traj_path + state + '/'

#             # The steps below are a modified version of Vision3D.fuse_point_clouds
#             pcl_vis = vis.Vision3D()
#             pc2 = o3d.io.read_point_cloud(state_path + 'pc_cam2.ply')
#             pc3 = o3d.io.read_point_cloud(state_path + 'pc_cam3.ply')
#             pc4 = o3d.io.read_point_cloud(state_path + 'pc_cam4.ply')
#             pc5 = o3d.io.read_point_cloud(state_path + 'pc_cam5.ply')

#             # combine the point clouds
#             pointcloud = o3d.geometry.PointCloud()
#             pointcloud.points = pc5.points
#             pointcloud.colors = pc5.colors
#             pointcloud.points.extend(pc2.points)
#             pointcloud.colors.extend(pc2.colors)
#             pointcloud.points.extend(pc3.points)
#             pointcloud.colors.extend(pc3.colors)
#             pointcloud.points.extend(pc4.points)
#             pointcloud.colors.extend(pc4.colors)

#             # remove statistical outliers
#             pointcloud, ind = pointcloud.remove_statistical_outlier(
#                 nb_neighbors=20, std_ratio=2.0
#             )  
            
#             # crop point cloud
#             pointcloud = pcl_vis.remove_stage_grippers(pointcloud) # Change ind_z_upper in Vision3D.remove_stage_grippers if you want the stage to be in frame
#             pointcloud = pcl_vis.remove_background(
#                 pointcloud, radius=0.15, center=np.array([0.6, -0.05, 0.3]) # change radius to 0.3 if you want the goal shape to be in frame
#             )
#             if s != len(states) - 1:
#                 action = np.load(traj_path + 'action5d_unnormalized' + str(s) + '.npy')
#             center = np.load(traj_path + 'pcl_center0' + '.npy')

#             # unscale and uncenter the state point cloud
#             state_unnormalized = np.asarray(pointcloud.points) * 0.1 + center
#             goal_unnormalized = None
            

#             rot = 360/n_rot_aug*j
#             state_aug, action_aug = augment_state_action(state_unnormalized, goal_unnormalized, center, action, rot, vis=False) 
#             pc_aug = o3d.geometry.PointCloud()
#             pc_aug.points = o3d.utility.Vector3dVector(state_aug)
#             pc_aug.colors = pointcloud.colors
#             vis1.add_geometry(pc_aug)


#             # Capture the Image 
#             img_arr = vis1.capture_screen_float_buffer(True)
#             numeric_part = ''.join(filter(str.isdigit, state))
#             filename = numeric_part.zfill(4) + '.jpg'
#             plt.imsave(save_path + '/' + filename, np.asarray(img_arr))
#             # vis1.run()
#             if s != len(states) - 1:
#                 actions_dict[filename] = action_aug.tolist()
#             vis1.remove_geometry(pc_aug)

#             print(j*n_traj+i,s)
#             s+=1

#         with open(save_path_actions,"w") as json_file:
#             json.dump(actions_dict, json_file)

# vis1.destroy_window()
