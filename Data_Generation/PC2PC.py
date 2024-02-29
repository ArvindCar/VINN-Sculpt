import numpy as np
import open3d as o3d
import sys
path_to_robomail = '/home/arvind/CMU/MAIL/RobomailPackages/robomail/'
sys.path.append(path_to_robomail)
import robomail.vision as vis
from matplotlib import pyplot as plt
import os
import json
from shapely.geometry import Point, Polygon
from skimage.color import rgb2lab
from augmentation_utils import *
import torch


def lab_color_crop(pcd_incoming):
        """
        Removes points from input point cloud based on color in LAB color space
        (currently thresholds for a green color)

        Args:
        pcd_incoming (o3d.geometry.PointCloud): input point cloud

        Returns:
        pcd (o3d.geometry.PointCloud): output point cloud with points removed
        """

        pcd = copy.deepcopy(pcd_incoming)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        lab_colors = rgb2lab(colors)  # covert colors to LAB color space

        a_idx = np.where(lab_colors[:, 1] < -5)
        l_idx = np.where(lab_colors[:, 0] > 0)
        b_idx = np.where(lab_colors[:, 2] > -5)

        indices = np.intersect1d(a_idx, np.intersect1d(l_idx, b_idx))

        pcd.points = o3d.utility.Vector3dVector(points[indices])
        pcd.colors = o3d.utility.Vector3dVector(colors[indices])

        return pcd



data_path = '/home/arvind/Clay_Data/Feb24_Discrete_Demos/Cone/Discrete/Train/' # Directory where all the collected data is
data_save_path = '/home/arvind/CMU/MAIL/VINN/VINN-Main/Data/PointClouds/Cone_all/'
n_traj = 10
n_rot_aug = 12
# save_path_goal = '/home/arvind/CMU/MAIL/VINN/VINN-Main/Data/PointClouds/X_all/New_Data/Goals'

for k in range(n_rot_aug*n_traj):

        j = k//n_traj
        i = k%n_traj
        save_path_img = data_save_path + 'run_pc_' + str(j*n_traj+i) + '/images' # Where you want this trajectory saved
        save_path_actions = data_save_path + 'run_pc_' + str(j*n_traj+i) + '/labels.json'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        traj_path = data_path + 'Trajectory' + str(i) + '/' 
        states = [state for state in os.listdir(traj_path) if state.startswith("Raw_State")] 
        states.sort()
        actions_dict = {}

        for s, state in enumerate(states): # Iterates through states within a given trajextory
            state_path = traj_path + state + '/'

            # The steps below are a modified version of Vision3D.fuse_point_clouds
            pcl_vis = vis.Vision3D()
            pc2 = o3d.io.read_point_cloud(state_path + 'pc_cam2.ply')
            pc3 = o3d.io.read_point_cloud(state_path + 'pc_cam3.ply')
            pc4 = o3d.io.read_point_cloud(state_path + 'pc_cam4.ply')
            pc5 = o3d.io.read_point_cloud(state_path + 'pc_cam5.ply')
            pointcloud = o3d.geometry.PointCloud()
            points = pcl_vis.fuse_point_clouds( pc2, pc3, pc4, pc5, vis=False, no_transformation=True)
            # pointcloud.points = o3d.utility.Vector3dVector(Points)
            # pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(Points), 1)))
            
            
            # uniformly sample 2048 points from each point cloud
            # points = np.asarray(pointcloud.points)
            idxs = np.random.randint(0, len(points), size=2048)
            points = points[idxs]
            # pointcloud.points = o3d.utility.Vector3dVector(points)
            # pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(points), 1)))
            # o3d.visualization.draw_geometries([pointcloud])


            if s != len(states) - 1:
                action = np.load(traj_path + 'action' + str(s) + '.npy')
            center = np.load(traj_path + 'pcl_center0' + '.npy')

            # points = points-center 
            # unscale and uncenter the state point cloud
            state_unnormalized = points * 0.1 + center
            goal_unnormalized = np.load(traj_path + 'new_goal_unnormalized.npy')

            
            rot = 360/n_rot_aug*j
            state_aug, action_aug, goal_aug = augment_state_action(state_unnormalized, goal_unnormalized, center, action, rot, vis=False) 
            state_aug = (state_aug-center) * 10
            goal_aug = (goal_aug - center) * 10
            # print(np.mean(state_aug, axis=0), np.mean(goal_aug, axis=0))
            # pointcloud.points = o3d.utility.Vector3dVector(state_aug)
            # pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(state_aug), 1)))
            # pointcloud.points.extend(o3d.utility.Vector3dVector(goal_aug))
            # pointcloud.colors.extend(o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(goal_aug), 1))))

            # o3d.visualization.draw_geometries([pointcloud])
            # Capture the PC
            
            numeric_part = ''.join(filter(str.isdigit, state))
            filename = numeric_part.zfill(4) + '.npy'
            np.save(save_path_img + '/' + filename, state_aug)
            if s==0:
                 np.save(save_path_img + '/goal', goal_aug)
            if s != len(states) - 1:
                actions_dict[filename] = action_aug.tolist()

            # print(actions_dict)
            print(j*n_traj+i,s)
            s+=1
        
        actions_keys = list(actions_dict.keys())
        actions_keys.sort()
        actions_dict = {i: actions_dict[i] for i in actions_keys}

        with open(save_path_actions,"w") as json_file:
            json.dump(actions_dict, json_file)