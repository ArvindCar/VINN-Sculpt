import math
import copy
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

'''
Contains necessary functions for rotation agumentation of the
clay datasets.
'''

def wrap_rz(original_rz):
    """
    We want rz to be between -90 to 90, so wrap around if outside these bounds due to symmetrical gripper.
    """
    wrapped_rz = (original_rz + 90) % 180 - 90
    return wrapped_rz

def line_3d_start_end(center, rz, length):
    """
    Given the center point, rotation and length of the line, generate one in o3d format for plotting
    """
    # convert rz to radians
    rz = np.radians(rz)
    dir_vec = np.array([np.cos(rz), np.sin(rz), 0])
    displacement = dir_vec * (0.5*length)
    start_point = center - displacement
    end_point = center + displacement
    points = np.array([start_point, end_point])
    # print("points: ", points)
    lines = np.array([[0,1]])
    return points, lines

def line_3d_point_set(points):
    """
    Given a list of list of points, convert to a list of points and create fully connected lines.
    """
    # print("points: ", points)
    new_points = []
    for elem in points:
        # print("\nelem: ", elem)
        for i in range(2):
            # print("\narr: ", elem[i])
            new_points.append(elem[i])
    # print("New Points: ", new_points)

    lines = np.array([[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7],
                      [1,2], [1,3], [1,4], [1,5] , [1,6], [1,7],
                      [2,3], [2,4], [2,5], [2,6], [2,7],
                      [3,4], [3,5], [3,6], [3,7],
                      [4,5], [4,6], [4,7],
                      [5,6], [5,7],
                      [6,7]])
    return new_points, lines

def augment_state_action(state, goal, center, action, rot, vis=False):
    '''
    Given a rotation about the z-axis, an action and state/goal/center
    pointclouds, return the state and goal pointclouds rotated by this 
    z-axis rotation as well as the action in the robot coordinate frame 
    rotated by this same augmentation angle. The process to do this is
    as follows:
        1.  center the state and goal point clouds by the center
        2.  apply rotation augmentation to the state and goal pointclouds
        3.  center the action w.r.t. the center of the state pointcloud to
            rotate about the same point
        4.  apply rotation augmentation to the centered action
        5.  uncenter the action back into the robot coordinate frame

    :param state: np.array of shape (n, 3) pointcloud
    :param goal:  np.array of shape (n, 3) pointcloud
    :param center: np.array of shape (3,) representing the center of the
                   point cloud in the robot frame (the point the rotation
                   augmentation is being applied about)
    :param action: np.array of shape (5,) representing the action in the
                   robot coordinate frame (x, y, z, rot_z, d_gripper)
    :param rot:    float representing the rotation augmentation in degrees
    :param vis:    bool representing whether the augmentation is for visualization
                   this distinction is necessary because there is a slight offset
                   for visualization to account for calibration error, but we
                   do not want this offset to be applied in our dataset creation 
                   as we want the augmented actions to remain in the Franka
                   corrdinat frame

    :return: augmented state, goal, action
    '''
    # speed up version apply augmentation to unnormalized point clouds
    if state is not None:
        state = state - center
        R = Rotation.from_euler('xyz', np.array([0, 0,  rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        
    if goal is not None:
        goal = goal - center
        R = Rotation.from_euler('xyz', np.array([0, 0,  rot]), degrees=True).as_matrix()        
        goal = R @ goal.T
        goal_aug = goal.T + center



    # # apply rotation augmentation to unnormalized point cloud
    # unnorm_pointcloud = o3d.geometry.PointCloud()
    # unnorm_pointcloud.points = o3d.utility.Vector3dVector(state)
    # unscale_pcl = copy.deepcopy(unnorm_pointcloud)
    # unscale_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (state.shape[0],1)))
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # R = mesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
    # unscale_pcl.rotate(R, center=center)
    # pcl_aug = np.asarray(unscale_pcl.points)

    # # apply rotation augmentation to unnormalized goal point cloud
    # unnorm_goal = o3d.geometry.PointCloud()
    # unnorm_goal.points = o3d.utility.Vector3dVector(goal)
    # unscale_goal = copy.deepcopy(unnorm_goal)
    # unscale_goal.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (goal.shape[0],1)))
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # R = mesh.get_rotation_matrix_from_xyz((0, 0, math.radians(rot)))
    # unscale_goal.rotate(R, center=center)
    # goal_aug = np.asarray(unscale_goal.points)

    # apply the action augmentation
    # need to apply a correction for calibration error if augmenting for visualization purposes
    if vis:
        new_action = action.copy()
        new_action[0:3] = new_action[0:3] + np.array([0.03, 0, 0])
        unit_circle_og_grasp = (new_action[0] - center[0], new_action[1] - center[1])
    # if augmenting for dataset creation, no need to apply correction
    else:
        unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])

    rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
    unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    rot_new =  rot_original + rot

    # NOTE: doesn't work slightly for grasps at edge of the clay (slight height variation)
    new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
    
    new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
    x = new_global_grasp[0]
    y = new_global_grasp[1]
    rz = action[3] + rot
    rz = wrap_rz(rz)
    action_aug = np.array([x, y, action[2], rz, action[4]])

    return (pcl_aug, action_aug, goal_aug) if goal is not None else (pcl_aug, action_aug) if state is not None else (action_aug)

def vis_grasp(state, next_state, action, offset=True):
    '''
    Given a state, next_state and action, visualize the grasp. This function
    assumes the state and next state pointclouds are not centered and unscaled
    and that the action is unnormalized and in the robot coordinate frame.

    :param state:  np.array of shape (n, 3) representing the state pointcloud
    :param next_state:  np.array of shape (n, 3) representing the next state 
                        pointcloud
    :param action:  np.array of shape (5,) representing the action in the robot 
                    coordinate frame
    :param offset:  bool representing whether to apply a slight offset to 
                    correct for calibration error - this should not be applied
                    when visualizing the augmentation, as the offset should
                    already be applied before rotating
    :return: None
    '''
    gripper_height = 0.06
    gripper_width = 0.02
    gripper_squeeze = 0.08

    # scale the action (multiply x,y,z,d by 10)
    action_scaled = action.copy()
    
    # adjusting for slight offset due to calibration for visualization purposes only
    if offset:
        action_scaled[0:3] = action[0:3] + np.array([0.03, 0, 0]) # adjusting for slight offset due to calibration 
    else:
        action_scaled[0:3] = action[0:3]

    len = gripper_squeeze 

    # get the points and lines for the action orientation visualization
    ctr = action_scaled[0:3]
    upper_ctr = ctr + np.array([0, 0, gripper_height]) 
    
    # apply rotation offset to account for the arbitrary zero rotation we picked on Franka setup
    rz = 90 + action_scaled[3] - 20 

    points, lines = line_3d_start_end(ctr, rz, len)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (lines.shape[0],1)))

    # get end points
    delta = gripper_squeeze - action_scaled[4]
    end_pts, _ = line_3d_start_end(ctr, rz, len - delta)
    top_end_pts, _ = line_3d_start_end(upper_ctr, rz, len - delta)

    # get the top points for the grasp (given gripper finger height)
    top_points, _ = line_3d_start_end(upper_ctr, rz, len)

    # gripper 1 
    g1_base_start, _ = line_3d_start_end(points[0], rz+90, gripper_width)
    g1_base_end, _ = line_3d_start_end(end_pts[0], rz+90, gripper_width)
    g1_top_start, _ = line_3d_start_end(top_points[0], rz+90, gripper_width)
    g1_top_end, _ = line_3d_start_end(top_end_pts[0], rz+90, gripper_width)
    g1_points, g1_lines = line_3d_point_set([g1_base_start, g1_base_end, g1_top_start, g1_top_end])

    # create oriented bounding box
    g1_test = o3d.geometry.OrientedBoundingBox()
    g1_bbox = g1_test.create_from_points(o3d.utility.Vector3dVector(g1_points))
    g1_idx = g1_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

    # create g1 
    g1 = o3d.geometry.LineSet()
    g1.points = o3d.utility.Vector3dVector(g1_points)
    g1.lines = o3d.utility.Vector2iVector(g1_lines)
    g1.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g1_lines.shape[0],1)))

    # gripper 2
    g2_base_start, _ = line_3d_start_end(points[1], rz+90, gripper_width)
    g2_base_end, _ = line_3d_start_end(end_pts[1], rz+90, gripper_width)
    g2_top_start, _ = line_3d_start_end(top_points[1], rz+90, gripper_width)
    g2_top_end, _ = line_3d_start_end(top_end_pts[1], rz+90, gripper_width)
    g2_points, g2_lines = line_3d_point_set([g2_base_start, g2_base_end, g2_top_start, g2_top_end])

    # create oriented bounding box
    g2_test = o3d.geometry.OrientedBoundingBox()
    g2_bbox = g2_test.create_from_points(o3d.utility.Vector3dVector(g2_points))
    g2_idx = g2_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(state))

    # create g2
    g2 = o3d.geometry.LineSet()
    g2.points = o3d.utility.Vector3dVector(g2_points)
    g2.lines = o3d.utility.Vector2iVector(g2_lines)
    g2.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (g2_lines.shape[0],1)))

    # create state next state pointclouds
    og_pcl = o3d.geometry.PointCloud()
    og_pcl.points = o3d.utility.Vector3dVector(state)
    og_colors = np.tile(np.array([0, 0, 1]), (state.shape[0],1))
    og_pcl.colors = o3d.utility.Vector3dVector(og_colors)

    g_pcl = o3d.geometry.PointCloud()
    g_pcl.points = o3d.utility.Vector3dVector(next_state)
    g_colors = np.tile(np.array([0, 1, 0]), (next_state.shape[0],1))
    g_pcl.colors = o3d.utility.Vector3dVector(g_colors)

    o3d.visualization.draw_geometries([og_pcl, g_pcl, line_set, g1, g2]) 