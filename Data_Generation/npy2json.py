from PIL import Image
import numpy as np
import open3d as o3d
from shapely.geometry import Point, Polygon
import robomail.vision as vis
from matplotlib import pyplot as plt
import os
import json


data_path = '/home/aesee/CMU/MAIL_Lab/Human Demos/Dec14_Human_Demos_Images/Dec14_Human_Demos_Raw/X/'
for i in range(10):
    save_path = data_path + 'run_' + str(i) + '/labels.json'
    traj_path = data_path + 'Trajectory' + str(i) + '/'
    actions = [action for action in os.listdir(traj_path) if action.startswith("action7d_unnormalized")]
    json_dict = {}
    for action in sorted(actions):
        numeric_part = ''.join(filter(str.isdigit, action[-7:]))
        img_name = numeric_part.zfill(4) + '.jpg'
        traj_data = np.load(traj_path + action).tolist()
        json_dict[img_name] = traj_data
        print(i, img_name, action)
    with open(save_path, "w") as json_file:
        json.dump(json_dict, json_file)



















































# vis1 = o3d.visualization.Visualizer()
# vis1.create_window()
# # data = np.load('/home/aesee/CMU/MAIL_Lab/Human Demos/State0/cam2_color.npy')
# # img = Image.fromarray(data)
# # img.save('/home/aesee/CMU/MAIL_Lab/Human Demos/State0/test.png')
# data_path = '/home/aesee/CMU/MAIL_Lab/Human Demos/Dec14_Human_Demos_Images/Dec14_Human_Demos_Raw/X/'
# for i in range(10):
#     save_path = data_path + 'run_' + str(i) + '/images'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     traj_path = data_path + 'Trajectory' + str(i) + '/'
#     states = [state for state in os.listdir(traj_path) if state.startswith("State")]
#     for state in states:
#         state_path = traj_path + state + '/'

#         pcl_vis = vis.Vision3D()
#         pc2 = o3d.io.read_point_cloud(state_path + 'pc_cam2.ply')
#         pc3 = o3d.io.read_point_cloud(state_path + 'pc_cam3.ply')
#         pc4 = o3d.io.read_point_cloud(state_path + 'pc_cam4.ply')
#         pc5 = o3d.io.read_point_cloud(state_path + 'pc_cam5.ply')

#         # combine the point clouds
#         pointcloud = o3d.geometry.PointCloud()
#         pointcloud.points = pc5.points
#         pointcloud.colors = pc5.colors
#         pointcloud.points.extend(pc2.points)
#         pointcloud.colors.extend(pc2.colors)
#         pointcloud.points.extend(pc3.points)
#         pointcloud.colors.extend(pc3.colors)
#         pointcloud.points.extend(pc4.points)
#         pointcloud.colors.extend(pc4.colors)

#         # crop point cloud
#         pointcloud, ind = pointcloud.remove_statistical_outlier(
#             nb_neighbors=20, std_ratio=2.0
#         )  # remove statistical outliers
#         pointcloud = pcl_vis.remove_stage_grippers(pointcloud)
#         pointcloud = pcl_vis.remove_background(
#             pointcloud, radius=0.17, center=np.array([0.6, -0.05, 0.3])
#         )

#         vis1.add_geometry(pointcloud)
#         # o3d.visualization.draw_geometries([pointcloud])
#         # vis1 = o3d.visualization.Visualizer()
#         # if i==0 and j==0:
#         #     vis1.create_window()
#         #     vis1.add_geometry(pointcloud)
#         # else:
#         #     vis1.update_geometry(pointcloud)
#         #     vis1.poll_events()
#         #     vis1.update_renderer()
        
#         print(i)
        
#         img_arr = vis1.capture_screen_float_buffer(True)
#         numeric_part = ''.join(filter(str.isdigit, state))
#         filename = numeric_part.zfill(4) + '.jpg'
#         plt.imsave(save_path + '/' + filename, np.asarray(img_arr))
#         vis1.remove_geometry(pointcloud)
#         # img = Image.fromarray(img_arr)
#         # img.save('/home/aesee/CMU/MAIL_Lab/Human Demos/test0.png')
# vis1.destroy_window()

        
