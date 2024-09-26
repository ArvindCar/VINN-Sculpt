import os
import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
import re
pointcloud = o3d.geometry.PointCloud()

def load_np_file(folder_path, file_name):
    np_file_path = os.path.join(folder_path, file_name)
    data = np.load(np_file_path)

    return data
def extract_number(file_name):
    return int(re.search(r'\d+', file_name).group())

def iterate_folders_and_load_np_files(root_dir, file_name):
    i=0
    x, y, z, theta, w, indices = [], [], [], [], [], []
    for folder_name in sorted(os.listdir(root_dir), key = extract_number):
        if i< 119:
            print(folder_name)
            i+=1
            folder_path = os.path.join(root_dir, folder_name)
            file_path = os.path.join(folder_path, file_name)
            action_file = open(file_path, 'r')
            act_dict = json.load(action_file)
            # print(act_dict['0000.npy'][0])
            # print(folder_name)
            state = '0000.jpg'
            x.append(act_dict[state][0])
            y.append(act_dict[state][1])
            z.append(act_dict[state][2])
            theta.append(act_dict[state][3])
            w.append(act_dict[state][4])
            indices.append(i-1)
    return x, y, z, theta, w, indices



# Specify the root directory containing the folders
root_directory = '/home/arvind/CMU/MAIL/VINN/VINN-Main/Data/Images/X_all/New_Data/X_img_new_shifted'

# Specify the name of the NumPy file you want to load from each folder
desired_file_name = "labels.json"

# Call the function to iterate through folders and load the specified NumPy file
x, y, z, theta, w, indices = iterate_folders_and_load_np_files(root_directory, desired_file_name)
scatter = plt.scatter(x, y, c=indices)

colorbar = plt.colorbar(scatter, label='Colorbar Label')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()
