import numpy as np
from augmentation_utils import *

'''
This is a simple script to sanity check that the rotation augmentation strategy is correct.
'''

trajectory_path = '/home/alison/Clay_Data/Trajectory_Data/No_Aug_Dec14_Human_Demos/X'
n_trajectories = 10

# dictionary with length of each trajectory (i.e. number of states -- number of actions = n_states - 1)
traj_dict = {0: 4,
             1: 7,
             2: 8,
             3: 8,
             4: 8,
             5: 8,
             6: 8,
             7: 9,
             8: 8,
             9: 8}

for i in range(2,n_trajectories):
    # NOTE: have the iteration through the rotation augmentations to apply here so that we can do it in order for each trajectory

    traj_path = trajectory_path + '/Trajectory' + str(i)


    # iterate through n_action:
    for i in range(traj_dict[i]-1):
        # load in a point cloud, goal and action
        state = np.load(traj_path + '/state' + str(i) + '.npy')
        goal_unnormalized = np.load(traj_path + '/goal.npy')
        action = np.load(traj_path + '/unnormalized_action' + str(i) + '.npy')
        center = np.load(traj_path + '/pcl_center' + str(i) + '.npy')

        # unscale and uncenter the state point cloud
        state_unnormalized = state * 0.1 + center

        # plot the original grasp action
        vis_grasp(state_unnormalized, goal_unnormalized, action, offset=False) #, center)

        for j in range(6):

            # augment the point cloud and grasp action
            rot = 60*j # [deg]
            
            state_aug, action_aug, goal_aug = augment_state_action(state_unnormalized, goal_unnormalized, center, action, rot, vis=False)

            # plot the augmented grasp action
            vis_grasp(state_aug, goal_aug, action_aug, offset=False) 


