import numpy as np
import torch
import glob
import random
from tqdm import tqdm
import json
import time
root = '/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/X_Datasets'
index = 0
runs = glob.glob(root+'/*')
random.shuffle(runs)
total = len(runs)
print(total)
img_tensors = []
for run_index in tqdm(range(total)):
    run = runs[run_index]
    action_file = open(run+'/labels.json', 'r')
    action_dict = json.load(action_file)
    for frame in action_dict:

        try:
            img = np.load(run+'/images/'+frame)
            img_tensor = torch.tensor(img)
        except:
            
            continue
        img_tensors.append(img_tensor.detach())
        index+=1
print(len(img_tensors))
# x = np.load(root + '0000.npy')
# img_tensor = torch.tensor(x)
# print(img_tensor.shape)

# print(img_tensor.detach().shape)