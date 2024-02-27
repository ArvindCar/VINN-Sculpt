import os
import numpy as np
import open3d as o3d
import torch
from torchvision import transforms as T


class CustomPointCloudAugmentation:
    def __init__(self, pointcloud_size=2048):
        self.pointcloud_size = pointcloud_size

    def __call__(self, pointcloud):
        pointcloud = self.shift_points(pointcloud)
        return pointcloud

    def shift_points(self, pointcloud, max_shift=0.0001):
        # Slightly shift each point in the point cloud
        shift_amounts = torch.rand_like(pointcloud) * 2 * max_shift - max_shift
        pointcloud = pointcloud + shift_amounts
        return pointcloud
    
def load_np_file(folder_path, file_name):
    np_file_path = os.path.join(folder_path, file_name)
    data = np.load(np_file_path)

    return data

def iterate_folders_and_load_np_files(root_dir, file_name):
    i=0
    aug = CustomPointCloudAugmentation()
    customAug = T.Compose([T.Lambda(lambda x: aug(x))])

    for folder_name in os.listdir(root_dir):
        i+=1
        folder_path = os.path.join(root_dir, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            data = load_np_file(folder_path, file_name)
            # if i!=1:
            #     pointcloud.points.extend(o3d.utility.Vector3dVector(data))    
            #     pointcloud.colors.extend(o3d.utility.Vector3dVector(np.tile((np.random.rand(3)), (len(data), 1))))
            # else:
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(data)
            pointcloud.colors = o3d.utility.Vector3dVector(np.tile((np.random.rand(3)), (len(data), 1)))
            pc_torch = torch.tensor(data)
            pc_aug = customAug(pc_torch)
            pc_torch = pc_aug.numpy()
            pointcloud.points.extend(o3d.utility.Vector3dVector(pc_torch))
            pointcloud.colors.extend(o3d.utility.Vector3dVector(np.tile((np.random.rand(3)), (len(pc_torch), 1))))
            o3d.visualization.draw_geometries([pointcloud])


# Specify the root directory containing the folders
root_directory = "/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/X_Datasets/"

# Specify the name of the NumPy file you want to load from each folder
desired_file_name = "images/0002.npy"

# Call the function to iterate through folders and load the specified NumPy file
iterate_folders_and_load_np_files(root_directory, desired_file_name)
