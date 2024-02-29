import os
import numpy as np
import open3d as o3d

pointcloud = o3d.geometry.PointCloud()

def load_np_file(folder_path, file_name):
    np_file_path = os.path.join(folder_path, file_name)
    data = np.load(np_file_path)

    return data

def iterate_folders_and_load_np_files(root_dir, file_name):
    i=0
    for folder_name in sorted(os.listdir(root_dir)):
        i+=1
        folder_path = os.path.join(root_dir, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            data = load_np_file(folder_path, file_name)
            if i!=1:
                pointcloud.points.extend(o3d.utility.Vector3dVector(data))    
                pointcloud.colors.extend(o3d.utility.Vector3dVector(np.tile((np.random.rand(3)), (len(data), 1))))
            else:
                pointcloud.points = o3d.utility.Vector3dVector(data)
                pointcloud.colors = o3d.utility.Vector3dVector(np.tile((np.random.rand(3)), (len(data), 1)))
            print(folder_name)
            o3d.visualization.draw_geometries([pointcloud])


# Specify the root directory containing the folders
root_directory = "/home/arvind/CMU/MAIL/VINN/VINN-Main/Data/PointClouds/Line"

# Specify the name of the NumPy file you want to load from each folder
desired_file_name = "images/goal.npy"

# Call the function to iterate through folders and load the specified NumPy file
iterate_folders_and_load_np_files(root_directory, desired_file_name)
