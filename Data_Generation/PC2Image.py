# import numpy as np
# import open3d as o3d
# import sys
# import autolab_core
# path_to_robomail = '/home/arvind/CMU/MAIL/RobomailPackages/robomail/'
# sys.path.append(path_to_robomail)
# import robomail.vision as vis
# from matplotlib import pyplot as plt
# import os
# import json
# from matplotlib.colors import Normalize

# from augmentation_utils import *

# vis1 = o3d.visualization.Visualizer()
# vis1.create_window()
# # print("hi")
# # time.sleep(10)
# data_path = '/home/arvind/Clay_Data/Feb24_Discrete_Demos/X/Discrete/Train/' # Directory where all the collected data is
# data_save_path = '/home/arvind/CMU/MAIL/VINN/VINN-Main/Data/Images/X_all/New_Data/'
# n_traj = 10
# n_rot_aug = 6

# for j in range(1):

#     for i in range(n_traj): # Change to number of trajectoriespr
#         save_path_img = data_save_path + 'run_' + str(j*n_traj+i) + '/images' # Where you want this trajectory saved
#         save_path_actions = data_save_path + 'run_' + str(j*n_traj+i) + '/labels.json'
#         if not os.path.exists(save_path_img):
#             os.makedirs(save_path_img)
#         traj_path = data_path + 'Trajectory' + str(i) + '/' 
#         states = [state for state in os.listdir(traj_path) if state.startswith("Raw_State")] 
#         states.sort()
#         actions_dict = {}

#         for s, state in enumerate(states): # Iterates through states within a given trajectory
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
#                 pointcloud, radius=0.25, center=np.array([0.6, -0.0, 0.22]) # change radius to 0.3 if you want the goal shape to be in frame
#             )
#             # o3d.visualization.draw_geometries([pointcloud]) # Uncomment if you want to visualize the PCs (For some reason it doesn't save when you visualize, so when you want to start saving, comment this line out)
#             if s != len(states) - 1:
#                 action = np.load(traj_path + 'action' + str(s) + '.npy')
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
#             plt.imsave(save_path_img + '/' + filename, np.asarray(img_arr))
#             # vis1.run()
#             if s != len(states) - 1:
#                 actions_dict[filename] = action_aug.tolist()
#             vis1.remove_geometry(pc_aug)

#             print(j*n_traj+i,state)
#             s+=1

#         actions_keys = list(actions_dict.keys())
#         actions_keys.sort()
#         actions_dict = {i: actions_dict[i] for i in actions_keys}
#         with open(save_path_actions,"w") as json_file:
#             json.dump(actions_dict, json_file)

# vis1.destroy_window()




import numpy as np
import open3d as o3d
import sys
path_to_robomail = '/home/arvind/CMU/MAIL/RobomailPackages/robomail/'
sys.path.append(path_to_robomail)
import robomail.vision as vis
from matplotlib import pyplot as plt
import os
import json
import time

from augmentation_utils import *

# vis1 = o3d.visualization.Visualizer()
# vis1.create_window()

data_path = '/home/arvind/Clay_Data/Feb24_Discrete_Demos/X/Discrete/Train/' # Directory where all the collected data is
data_save_path = '/home/arvind/CMU/MAIL/VINN/VINN-Main/Data/Images/X_all/New_Data/X_img_new_shifted/'
n_traj = 10
n_rot_aug = 6

for j in range(n_rot_aug):

    for i in range(n_traj): # Change to number of trajectories
        save_path_img = data_save_path + 'run_' + str(j*n_traj+i) + '/images' # Where you want this trajectory saved
        save_path_actions = data_save_path + 'run_' + str(j*n_traj+i) + '/labels.json'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        traj_path = data_path + 'Trajectory' + str(i) + '/' 
        states = [state for state in os.listdir(traj_path) if state.startswith("state")] 
        states.sort()
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
            if s != len(states) - 1:
                action = np.load(traj_path + 'action' + str(s) + '.npy')
            center = np.load(traj_path + 'pcl_center0' + '.npy')

            # # unscale and uncenter the state point cloud
            # state_unnormalized = np.asarray(pointcloud.points) * 0.1 + center
            # goal_unnormalized = None
            state_unnormalized = None
            goal_unnormalized = None            

            rot = 360/n_rot_aug*j
            action_aug = augment_state_action(state_unnormalized, goal_unnormalized, center, action, rot, vis=False, shift = [-0.02, 0.02])
            # pc_aug = o3d.geometry.PointCloud()
            # pc_aug.points = o3d.utility.Vector3dVector(state_aug)
            # pc_aug.colors = pointcloud.colors
            # vis1.add_geometry(pc_aug)


            # Capture the Image 
            # img_arr = vis1.capture_screen_float_buffer(True)
            numeric_part = ''.join(filter(str.isdigit, state))
            filename = numeric_part.zfill(4) + '.jpg'
            # plt.imsave(save_path + '/' + filename, np.asarray(img_arr))
            # vis1.run()
            if s != len(states) - 1:
                actions_dict[filename] = action_aug.tolist()
            # vis1.remove_geometry(pc_aug)

            print(j*n_traj+i,s)
            s+=1
        actions_keys = list(actions_dict.keys())
        actions_keys.sort()
        actions_dict = {i: actions_dict[i] for i in actions_keys}


        with open(save_path_actions,"w") as json_file:
            json.dump(actions_dict, json_file)

# vis1.destroy_window()