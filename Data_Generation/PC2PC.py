import numpy as np
import open3d as o3d
import robomail.vision as vis
from matplotlib import pyplot as plt
import os
import json
from shapely.geometry import Point, Polygon
from skimage.color import rgb2lab
from augmentation_utils import *
import torch

# vis1 = o3d.visualization.Visualizer()
# vis1.create_window()
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



data_path = '/home/aesee/CMU/MAIL_Lab/Human Demos/Dec14_Human_Demos_Images/Dec14_Human_Demos_Raw/X/' # Directory where all the collected data is
n_traj = 10
n_rot_aug = 6

for j in range(n_rot_aug):

    for i in range(n_traj): # Change to number of trajectories
        save_path = data_path + 'run_pc_' + str(j*n_traj+i) + '/images' # Where you want this trajectory saved
        save_path_actions = data_path + 'run_pc_' + str(j*n_traj+i) + '/labels.json'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        traj_path = data_path + 'Trajectory' + str(i) + '/' 
        states = [state for state in os.listdir(traj_path) if state.startswith("State")] 
        actions_dict = {}

        for s, state in enumerate(states): # Iterates through states within a given trajextory
            state_path = traj_path + state + '/'

            # # The steps below are a modified version of Vision3D.fuse_point_clouds
            # pcl_vis = vis.Vision3D()
            # pc2 = o3d.io.read_point_cloud(state_path + 'pc_cam2.ply')
            # pc3 = o3d.io.read_point_cloud(state_path + 'pc_cam3.ply')
            # pc4 = o3d.io.read_point_cloud(state_path + 'pc_cam4.ply')
            # pc5 = o3d.io.read_point_cloud(state_path + 'pc_cam5.ply')

            # # combine the point clouds
            # pointcloud = o3d.geometry.PointCloud()
            # pointcloud.points = pc5.points
            # pointcloud.colors = pc5.colors
            # pointcloud.points.extend(pc2.points)
            # pointcloud.colors.extend(pc2.colors)
            # pointcloud.points.extend(pc3.points)
            # pointcloud.colors.extend(pc3.colors)
            # pointcloud.points.extend(pc4.points)
            # pointcloud.colors.extend(pc4.colors)

            # # remove statistical outliers
            # pointcloud, ind = pointcloud.remove_statistical_outlier(
            #     nb_neighbors=20, std_ratio=2.0
            # )  
            
            # # crop point cloud
            # pointcloud = pcl_vis.remove_stage_grippers(pointcloud) # Change ind_z_upper in Vision3D.remove_stage_grippers if you want the stage to be in frame
            # pointcloud = pcl_vis.remove_background(
            #     pointcloud, radius=0.15, center=np.array([0.6, -0.05, 0.3]) # change radius to 0.3 if you want the goal shape to be in frame
            # )

            # # color thresholding
            # pointcloud = lab_color_crop(pointcloud)
            # pointcloud, ind = pointcloud.remove_statistical_outlier(
            #     nb_neighbors=20, std_ratio=2.0
            # )

            # # get shape of clay base
            # pointcloud.estimate_normals()
            # downpdc = pointcloud.voxel_down_sample(voxel_size=0.0025)
            # downpdc_points = np.asarray(downpdc.points)
            # # polygon_indices = np.where(downpdc_points[:,2] < 0.236) # PREVIOUS BEFORE 8/29
            # polygon_indices = np.where(downpdc_points[:, 2] < 0.22)

            # # polygon_indices = np.where(downpdc_points[:,2] < 0.234)
            # polygon_pcl = o3d.geometry.PointCloud()
            # polygon_pcl.points = o3d.utility.Vector3dVector(downpdc_points[polygon_indices])

            # # generate a 2d grid of points for the base
            # base_plane = []
            # minx, maxx = np.amin(downpdc_points[:, 0]), np.amax(downpdc_points[:, 0])
            # miny, maxy = np.amin(downpdc_points[:, 1]), np.amax(downpdc_points[:, 1])
            # minz, maxz = np.amin(downpdc_points[:, 2]), np.amax(downpdc_points[:, 2])
            # x_vals = np.linspace(minx, maxx, 50)
            # y_vals = np.linspace(miny, maxy, 50)
            # xx, yy = np.meshgrid(
            #     x_vals, y_vals
            # )  # create grid that covers full area of 2d polygon
            # # z = 0.234 # height of the clay base

            # z = 0.21
            # # z = 0.232 # PREVIOUS BEFORE 8/29
            # zz = np.ones(len(xx.flatten())) * z
            # points = np.vstack((xx.flatten(), yy.flatten(), zz)).T

            # grid_cloud = o3d.geometry.PointCloud()
            # grid_cloud.points = o3d.utility.Vector3dVector(points)

            # # crop shape of clay base out of 2d grid of points
            # polygon_coords = np.asarray(polygon_pcl.points)[:, 0:2]
            # polygon = Polygon(polygon_coords)
            # mask = [
            #     polygon.contains(Point(x, y))
            #     for x, y in np.asarray(grid_cloud.points)[:, 0:2]
            # ]
            # cropped_grid = np.asarray(grid_cloud.points)[:, 0:2][mask]
            # zs = np.ones(len(cropped_grid)) * z
            # cropped_grid = np.concatenate(
            #     (cropped_grid, np.expand_dims(zs, axis=1)), axis=1
            # )

            # base_cloud = o3d.geometry.PointCloud()
            # base_cloud.points = o3d.utility.Vector3dVector(cropped_grid)

            # # add top part of clay to new base
            # base_cloud.points.extend(downpdc.points)
            # cropped_plane, ind = base_cloud.remove_statistical_outlier(
            #     nb_neighbors=20, std_ratio=2.0
            # )

            # base_cloud.colors = o3d.utility.Vector3dVector(
            #     np.tile(np.array([0, 0, 1]), (len(base_cloud.points), 1))
            # )

            # # uniformly sample 2048 points from each point cloud
            # points = np.asarray(base_cloud.points)
            # idxs = np.random.randint(0, len(points), size=2048)
            # points = points[idxs]
            # # pointcloud.points = o3d.utility.Vector3dVector(points)
            # # pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(points), 1)))
            # # o3d.visualization.draw_geometries([pointcloud])


            if s != len(states) - 1:
                action = np.load(traj_path + 'action5d_unnormalized' + str(s) + '.npy')
            center = np.load(traj_path + 'pcl_center0' + '.npy')

            # points = points-center
            # # unscale and uncenter the state point cloud
            # state_unnormalized = points * 0.1 + center
            # goal_unnormalized = None

            
            rot = 360/n_rot_aug*j
            action_aug = augment_state_action(None, None, center, action, rot, vis=False)  # this needs to be deleted
            # state_aug, action_aug = augment_state_action(state_unnormalized, goal_unnormalized, center, action, rot, vis=False) 
            # state_aug = (state_aug-center)*10
            # print(np.mean(state_aug, axis=0))
            # pointcloud.points = o3d.utility.Vector3dVector(state_aug)
            # pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(state_aug), 1)))
            # o3d.visualization.draw_geometries([pointcloud])
            # Capture the PC
            
            # img_arr = vis1.capture_screen_float_buffer(True)
            numeric_part = ''.join(filter(str.isdigit, state))
            filename = numeric_part.zfill(4) + '.npy'
            # np.save(save_path + '/' + filename, state_aug)

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