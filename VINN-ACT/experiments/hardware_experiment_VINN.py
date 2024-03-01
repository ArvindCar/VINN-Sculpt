import os
import cv2
import time
import torch
import queue
import threading
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import robomail.vision as vis
from frankapy import FrankaArm
from distance_metrics import *
from scipy.spatial.transform import Rotation
from x_pc_nn_sculpt import VINN_Img


'''
This is the generic script for the clay hardware experiments. It will save all the necessary information to
document each experiment. This includes the following:
    - RGB image from each camera
    - goal point cloud
    - state point clouds
    - number of actions to completion
    - real-world time to completion
    - chamfer distance between final state and goal
    - earth mover's distance between final state and goal
    - video from camera 6 recording the entire experimental run
'''

def goto_grasp(fa, x, y, z, rx, ry, rz, d):
	"""
	Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
	This function was designed to be used for clay moulding, but in practice can be applied to any task.

	:param fa:  franka robot class instantiation
	"""
	pose = fa.get_pose()
	starting_rot = pose.rotation
	orig = Rotation.from_matrix(starting_rot)
	orig_euler = orig.as_euler('xyz', degrees=True)
	rot_vec = np.array([rx, ry, rz])
	new_euler = orig_euler + rot_vec
	r = Rotation.from_euler('xyz', new_euler, degrees=True)
	pose.rotation = r.as_matrix()
	pose.translation = np.array([x, y, z])

	fa.goto_pose(pose)
	fa.goto_gripper(d, force=60.0)
	time.sleep(3)

def experiment_loop(fa, cam2, cam3, cam4, cam5, pcl_vis, save_path, goal, done_queue):
    '''
    '''
    # define observation pose
    pose = fa.get_pose()
    observation_pose = np.array([0.6, 0, 0.325])
    pose.translation = observation_pose
    fa.goto_pose(pose)

    # define the action space limits for unnormalization
    a_mins5d = np.array([0.55, -0.035, 0.19, -90, 0.005])
    a_maxs5d = np.array([0.63, 0.035, 0.25, 90, 0.05])
    
    # initialize the n_actions counter
    n_action = 0

    # get the starting time
    start_time = time.time()

    # get the observation state
    rgb2, _, pc2, _ = cam2._get_next_frame()
    rgb3, _, pc3, _ = cam3._get_next_frame()
    rgb4, _, pc4, _ = cam4._get_next_frame()
    rgb5, _, pc5, _ = cam5._get_next_frame()
    pointcloud = pcl_vis.fuse_point_clouds(pc2, pc3, pc4, pc5, vis=False)

    
    # TODO: Center the PC
    pointcloud = pointcloud - np.mean(pointcloud, axis = 0)

    # save the point clouds from each camera
    o3d.io.write_point_cloud(save_path + '/cam2_pcl0.ply', pc2)
    o3d.io.write_point_cloud(save_path + '/cam3_pcl0.ply', pc3)
    o3d.io.write_point_cloud(save_path + '/cam4_pcl0.ply', pc4)
    o3d.io.write_point_cloud(save_path + '/cam5_pcl0.ply', pc5)

    # save observation
    np.save(save_path + '/pcl0.npy', pointcloud)
    cv2.imwrite(save_path + '/rgb2_state0.jpg', rgb2)
    cv2.imwrite(save_path + '/rgb3_state0.jpg', rgb3)
    cv2.imwrite(save_path + '/rgb4_state0.jpg', rgb4)
    cv2.imwrite(save_path + '/rgb5_state0.jpg', rgb5)

    # get the distance metrics between the point cloud and goal
    cd = chamfer(pointcloud, goal)
    earthmovers = emd(pointcloud, goal)
    hausdorff_dist = hausdorff(pointcloud, goal)
    print("\nChamfer Distance: ", cd)
    print("Earth Mover's Distance: ", earthmovers)
    print("Hausdorff Distance: ", hausdorff_dist)

    root_dir = '/home/arvind/VINN-Sculpt/VINN-Sculpt/VINN-ACT/'
    chkpts_dir = 'chkpts/BYOL_100_X_batch_30.pt'
    baseline = VINN_Img(root_dir, chkpts_dir)

    exec_times = []
    process_dict = []
    for i in range(12): # maximum number of actions allowed (CAN BE ADJUSTED!)
        print("Action:",i)
        start_exec = time.time()
        # generate the next action given the observation and goal and convert to the robot's coordinate frame
        pred_action_sequence = [] # TODO: fill this in with respective action sequence prediction model
        pred_action = baseline.next_action(pc2, pc3, pc4, pc5, goal) # TODO: include a parameter to execute N steps before replanning
        end_exec = time.time()
        exec_times.append(end_exec - start_exec)
        # unnorm_a = (pred_action + 1)/2.0 # NOTE: this step is for the model to output actions in the range [-1, 1], if the model outputs actions in the range [0, 1], this step is not necessary
        # unnorm_a = unnorm_a * (a_maxs5d - a_mins5d) + a_mins5d
        unnorm_a = pred_action
        # execute the unnormalized action
        goto_grasp(fa, unnorm_a[0], unnorm_a[1], unnorm_a[2], 0, 0, unnorm_a[3], unnorm_a[4])
        n_action+=1

        # wait here
        time.sleep(3)

        # open the gripper
        fa.open_gripper(block=True)
        # time.sleep(2)

        # move to observation pose
        pose.translation = observation_pose
        fa.goto_pose(pose)

        # get the observation state
        rgb2, _, pc2, _ = cam2._get_next_frame()
        rgb3, _, pc3, _ = cam3._get_next_frame()
        rgb4, _, pc4, _ = cam4._get_next_frame()
        rgb5, _, pc5, _ = cam5._get_next_frame()
        pointcloud = pcl_vis.fuse_point_clouds(pc2, pc3, pc4, pc5, vis=False)

        # save the point clouds from each camera
        o3d.io.write_point_cloud(save_path + '/cam2_pcl' + str(i+1) + '.ply', pc2)
        o3d.io.write_point_cloud(save_path + '/cam3_pcl' + str(i+1) + '.ply', pc3)
        o3d.io.write_point_cloud(save_path + '/cam4_pcl' + str(i+1) + '.ply', pc4)
        o3d.io.write_point_cloud(save_path + '/cam5_pcl' + str(i+1) + '.ply', pc5)

        # save observation
        np.save(save_path + '/pcl' + str(i+1) + '.npy', pointcloud)
        cv2.imwrite(save_path + '/rgb2_state' + str(i+1) + '.jpg', rgb2)
        cv2.imwrite(save_path + '/rgb3_state' + str(i+1) + '.jpg', rgb3)
        cv2.imwrite(save_path + '/rgb4_state' + str(i+1) + '.jpg', rgb4)
        cv2.imwrite(save_path + '/rgb5_state' + str(i+1) + '.jpg', rgb5)

        # get the distance metrics between the point cloud and goal
        cd = chamfer(pointcloud, goal)
        earthmovers = emd(pointcloud, goal)
        hausdorff_dist = hausdorff(pointcloud, goal)
        print("\nChamfer Distance: ", cd)
        print("Earth Mover's Distance: ", earthmovers)
        print("Hausdorff Distance: ", hausdorff_dist)
        curr_dict = {'n_action': i, 'chamfer_distance': cd, 'earth_movers_distance': earthmovers, 'Hausdorff Distance:': hausdorff_dist}
        process_dict.append(curr_dict)
        # exit loop early if the goal is reached
        if earthmovers < 0.06 or cd < 0.06:
            break

        # alternate break scenario --> if the past 3 actions have not resulted in a decent change in the emd or cd, break
    
    # completed the experiment, send the message to the video recording loop
    done_queue.put("Done!")

    # get the ending time
    end_time = time.time()
    
    # create and save a dictionary of the experiment results
    results_dict = {'n_actions': n_action, 'time_to_completion': end_time - start_time, 'chamfer_distance': cd, 'earth_movers_distance': earthmovers, 'Hausdorff Distance:': hausdorff_dist}
    with open(save_path + '/results.txt', 'w') as f:
        f.write(str(results_dict))
    with open(save_path + '/progress.txt', 'w') as f:
        f.write(str(process_dict))
    with open(save_path + '/exec_times.txt', 'w') as f:
        f.write(str(exec_times))

# VIDEO THREAD
def video_loop(cam_pipeline, save_path, done_queue):
    '''
    '''
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    out = cv2.VideoWriter(save_path + '/video.avi', forcc, 30.0, (848, 480))

    frame_save_counter = 0
    # record until main loop is complete
    while done_queue.empty():
        frames = cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # crop and rotate the image to just show elevated stage area
        cropped_image = color_image[320:520,430:690,:]
        rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # save frame approx. every 10 seconds
        if frame_save_counter % 300 == 0:
            cv2.imwrite(save_path + '/external_rgb' + str(frame_save_counter) + '.jpg', rotated_image)
        frame_save_counter += 1
        out.write(rotated_image)
    
    cam_pipeline.stop()
    out.release()

if __name__ == '__main__':
    # -------------------------------------------------------------------
    # ---------------- Experimental Parameters to Define ----------------
    # -------------------------------------------------------------------
    exp_num = 1
    goal_shape = 'X' # 'cone' or 'line' or 'X' or 'Y' or 'cylinder
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------


    exp_save = '/home/arvind/VINN-Sculpt/Experiments/Exp' + str(exp_num)

    # check to make sure the experiment number is not already in use, if it is, increment the number to ensure no save overwrites
    while os.path.exists(exp_save):
        exp_num += 1
        exp_save = '/home/arvind/VINN-Sculpt/Experiments/Exp' + str(exp_num)

    # make the experiment folder
    os.mkdir(exp_save)

    # initialize the robot and reset joints
    fa = FrankaArm()
    fa.reset_joints()

    # initialize the cameras
    cam2 = vis.CameraClass(2)
    cam3 = vis.CameraClass(3)
    cam4 = vis.CameraClass(4)
    cam5 = vis.CameraClass(5)
    
    # initialize camera 6 pipeline
    W = 1280
    H = 800
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('152522250441')
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    # initialize the 3D vision code
    pcl_vis = vis.Vision3D()    

    # load in the goal and save to the experiment folder
    goal = np.load('/home/arvind/sculpt-act_baseline/TargetPC/X_final.npy')
    np.save(exp_save + '/goal.npy', goal)

    # initialize the threads
    done_queue = queue.Queue()

    main_thread = threading.Thread(target=experiment_loop, args=(fa, cam2, cam3, cam4, cam5, pcl_vis, exp_save, goal, done_queue))
    video_thread = threading.Thread(target=video_loop, args=(pipeline, exp_save, done_queue))

    main_thread.start()
    video_thread.start()