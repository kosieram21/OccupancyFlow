import os
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
from subprocess import call
from tqdm import tqdm

PIXELS_PER_METER = 3.2
SDC_X_IN_GRID = 112 #512
SDC_Y_IN_GRID = 112 #512

GRID_SIZE = 224 #1024
PADDING = 0

DPI = 1
IMG_SIZE = GRID_SIZE / DPI

roadgraph_features = {
    'roadgraph_samples/dir': 'float',
    'roadgraph_samples/id': 'int',
    'roadgraph_samples/type': 'int',
    'roadgraph_samples/valid': 'int',
    'roadgraph_samples/xyz': 'float',
}

state_features = {
    'state/id': 'float',
    'state/type': 'float',
    'state/is_sdc': 'int',
    'state/tracks_to_predict': 'int',
    'state/current/bbox_yaw': 'float',
    'state/current/height': 'float',
    'state/current/length': 'float',
    'state/current/timestamp_micros': 'int',
    'state/current/valid': 'int',
    'state/current/vel_yaw': 'float',
    'state/current/velocity_x': 'float',
    'state/current/velocity_y': 'float',
    'state/current/width': 'float',
    'state/current/x': 'float',
    'state/current/y': 'float',
    'state/current/z': 'float',
    'state/future/bbox_yaw': 'float',
    'state/future/height': 'float',
    'state/future/length': 'float',
    'state/future/timestamp_micros': 'int',
    'state/future/valid': 'int',
    'state/future/vel_yaw': 'float',
    'state/future/velocity_x': 'float',
    'state/future/velocity_y': 'float',
    'state/future/width': 'float',
    'state/future/x': 'float',
    'state/future/y': 'float',
    'state/future/z': 'float',
    'state/past/bbox_yaw': 'float',
    'state/past/height': 'float',
    'state/past/length': 'float',
    'state/past/timestamp_micros': 'int',
    'state/past/valid': 'int',
    'state/past/vel_yaw': 'float',
    'state/past/velocity_x': 'float',
    'state/past/velocity_y': 'float',
    'state/past/width': 'float',
    'state/past/x': 'float',
    'state/past/y': 'float',
    'state/past/z': 'float',
}

traffic_light_features = {
    'traffic_light_state/current/state': 'int',
    'traffic_light_state/current/valid': 'int',
    'traffic_light_state/current/x': 'float',
    'traffic_light_state/current/y': 'float',
    'traffic_light_state/current/z': 'float',
    'traffic_light_state/past/state': 'int',
    'traffic_light_state/past/valid': 'int',
    'traffic_light_state/past/x': 'float',
    'traffic_light_state/past/y': 'float',
    'traffic_light_state/past/z': 'float',
    'traffic_light_state/future/state': 'int',
    'traffic_light_state/future/valid': 'int',
    'traffic_light_state/future/x': 'float',
    'traffic_light_state/future/y': 'float',
    'traffic_light_state/future/z': 'float',
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

roadgraph_transforms = {
    'roadgraph_samples/dir':
        lambda x : np.reshape(x,(20000,3)),
    'roadgraph_samples/id':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/type':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/valid':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/xyz':
        lambda x : np.reshape(x,(20000,3)),
}

state_transforms = {
    'state/id':
        lambda x : np.reshape(x,(128,)),
    'state/type':
        lambda x : np.reshape(x,(128,)),
    'state/is_sdc':
        lambda x : np.reshape(x,(128,)),
    'state/tracks_to_predict':
        lambda x : np.reshape(x,(128,)),
    'state/current/bbox_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/height':
        lambda x : np.reshape(x,(128,1)),
    'state/current/length':
        lambda x : np.reshape(x,(128,1)),
    'state/current/timestamp_micros':
        lambda x : np.reshape(x,(128,1)),
    'state/current/valid':
        lambda x : np.reshape(x,(128,1)),
    'state/current/vel_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/width':
        lambda x : np.reshape(x,(128,1)),
    'state/current/x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/z':
        lambda x : np.reshape(x,(128,1)),
    'state/future/bbox_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/height':
        lambda x : np.reshape(x,(128,80)),
    'state/future/length':
        lambda x : np.reshape(x,(128,80)),
    'state/future/timestamp_micros':
        lambda x : np.reshape(x,(128,80)),
    'state/future/valid':
        lambda x : np.reshape(x,(128,80)),
    'state/future/vel_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/width':
        lambda x : np.reshape(x,(128,80)),
    'state/future/x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/z':
        lambda x : np.reshape(x,(128,80)),
    'state/past/bbox_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/height':
        lambda x : np.reshape(x,(128,10)),
    'state/past/length':
        lambda x : np.reshape(x,(128,10)),
    'state/past/timestamp_micros':
        lambda x : np.reshape(x,(128,10)),
    'state/past/valid':
        lambda x : np.reshape(x,(128,10)),
    'state/past/vel_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/width':
        lambda x : np.reshape(x,(128,10)),
    'state/past/x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/z':
        lambda x : np.reshape(x,(128,10)),
}

traffic_light_transforms = {
    'traffic_light_state/current/state':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/valid':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/x':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/y':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/z':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/past/state':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/valid':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/x':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/y':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/z':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/future/state':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/valid':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/x':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/y':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/z':
        lambda x : np.reshape(x,(80,16)),
}

features_transforms = {}
features_transforms.update(roadgraph_transforms)
features_transforms.update(state_transforms)
features_transforms.update(traffic_light_transforms)

road_label = {1:'LaneCenter-Freeway', 2:'LaneCenter-SurfaceStreet', 3:'LaneCenter-BikeLane', 6:'RoadLine-BrokenSingleWhite',
              7:'RoadLine-SolidSingleWhite', 8:'RoadLine-SolidDoubleWhite', 9:'RoadLine-BrokenSingleYellow', 10:'RoadLine-BrokenDoubleYellow', 
              11:'Roadline-SolidSingleYellow', 12:'Roadline-SolidDoubleYellow', 13:'RoadLine-PassingDoubleYellow', 15:'RoadEdgeBoundary', 
              16:'RoadEdgeMedian', 17:'StopSign', 18:'Crosswalk', 19:'SpeedBump'}

road_line_map = {1:['xkcd:grey', 'solid', 14], 2:['xkcd:grey', 'solid', 14], 3:['xkcd:grey', 'solid', 10], 6:['w', 'dashed', 2], 
                 7:['w', 'solid', 2], 8:['w', 'solid', 2], 9:['xkcd:yellow', 'dashed', 4], 10:['xkcd:yellow', 'dashed', 2], 
                 11:['xkcd:yellow', 'solid', 2], 12:['xkcd:yellow', 'solid', 3], 13:['xkcd:yellow', 'dotted', 1.5], 15:['y', 'solid', 4.5], 
                 16:['y', 'solid', 4.5], 17:['r', '.', 40], 18:['b', 'solid', 13], 19:['xkcd:orange', 'solid', 13]}

light_label = {0:'Unknown', 1:'Arrow_Stop', 2:'Arrow_Caution', 3:'Arrow_Go', 4:'Stop', 
               5:'Caution', 6:'Go', 7:'Flashing_Stop', 8:'Flashing_Caution'}

light_state_map = {0:'k', 1:'r', 2:'y', 3:'g', 4:'r', 5:'y', 6:'g', 7:'r', 8:'y'}

def transform(feature):
    transform = features_transforms
    keys = transform.keys()
    for key in keys:
        func = transform[key]
        feat = feature[key]
        feature[key] = func(feat)
    return feature

def create_idx(tfrecord_dir, idx_dir):
    os.makedirs(idx_dir, exist_ok=True)
    for tfrecord in tqdm(glob.glob(tfrecord_dir+'/*')):
        idxname = idx_dir + '/' + tfrecord.split('/')[-1]
        call(["tfrecord2idx", tfrecord, idxname])

def WaymoDataset(tfrecord_dir, idx_dir):
    tfrecord_pattern = tfrecord_dir+'/{}'
    index_pattern = idx_dir+'/{}'
    fnlist = os.listdir(tfrecord_pattern.split('{}')[0])
    splits = {fn: 1/len(fnlist) for fn in fnlist}
    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description=features_description, transform=transform, infinite=False)
    return dataset

def rotate_points_around_origin(points, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points, rotation_matrix.T)

def normalize_about_sdc(points, data):
    # get self-driving car (sdc) current position and yaw
    sdc_indices = np.where(data['state/is_sdc'] == 1)
    sdc_xy = np.column_stack((data['state/current/x'][sdc_indices].flatten(),
                              data['state/current/y'][sdc_indices].flatten()))
    sdc_bbox_yaw = data['state/current/bbox_yaw'][sdc_indices].item()

    # provide translational and rotatinal invarience (with respect to sdc)
    centered_points = points - sdc_xy
    angle = math.pi / 2 - sdc_bbox_yaw
    centered_and_rotated_points = rotate_points_around_origin(centered_points, angle)

    return centered_and_rotated_points

def denormalize_from_sdc(centered_and_rotated_points, data):
    # get self-driving car (sdc) current position and yaw
    sdc_indices = np.where(data['state/is_sdc'] == 1)
    sdc_xy = np.column_stack((data['state/current/x'][sdc_indices].flatten(),
                              data['state/current/y'][sdc_indices].flatten()))
    sdc_bbox_yaw = data['state/current/bbox_yaw'][sdc_indices].item()

    # Reverse the rotation and translation
    angle = -(math.pi / 2 - sdc_bbox_yaw)
    centered_points = rotate_points_around_origin(centered_and_rotated_points, angle)
    points = centered_points + sdc_xy
    
    return points

def get_image_coordinates(world_points):
    scale = np.array([PIXELS_PER_METER, -PIXELS_PER_METER])
    offset = np.array([SDC_X_IN_GRID, SDC_Y_IN_GRID])
    image_points = np.round(world_points * scale) + offset
    return image_points

def get_world_coordinates(image_points):
    offset = np.array([SDC_X_IN_GRID, SDC_Y_IN_GRID])
    scale = np.array([PIXELS_PER_METER, -PIXELS_PER_METER])
    world_points = (image_points - offset) / scale
    return world_points

def get_fov_mask(points):
    fov_mask = np.logical_and.reduce([
        points[:, 0] >= -PADDING,
        points[:, 0] < GRID_SIZE + PADDING,
        points[:, 1] >= -PADDING,
        points[:, 1] < GRID_SIZE + PADDING
    ])

    return fov_mask

def collate_agent_trajectories(data):
    past_positions = np.stack((data['state/past/x'], data['state/past/y']), axis=-1)
    current_position = np.stack((data['state/current/x'], data['state/current/y']), axis=-1)
    observed_positions = np.concatenate((past_positions, current_position), axis=1)
    print(observed_positions[0])

    max_agents, timesteps, xy = observed_positions.shape
    observed_positions = observed_positions.reshape(-1, xy)
    
    centered_and_rotated_observed_positions = normalize_about_sdc(observed_positions, data)
    centered_and_rotated_image_observed_positions = get_image_coordinates(centered_and_rotated_observed_positions)
    fov_mask = get_fov_mask(centered_and_rotated_image_observed_positions)

    observed_positions = centered_and_rotated_observed_positions.reshape(max_agents, timesteps, xy)
    print(observed_positions[0])
    fov_mask = fov_mask.reshape(max_agents, timesteps)

    past_states = np.stack((data['state/past/bbox_yaw'],
                            data['state/past/velocity_x'], data['state/past/velocity_y'], data['state/past/vel_yaw'],
                            data['state/past/width'], data['state/past/length'],
                            data['state/past/timestamp_micros'] / 1000000), axis=-1)
    past_states_valid = data['state/past/valid'] > 0.

    current_states = np.stack((data['state/current/bbox_yaw'],
                               data['state/current/velocity_x'], data['state/current/velocity_y'], data['state/current/vel_yaw'],
                               data['state/current/width'], data['state/current/length'],
                               data['state/current/timestamp_micros'] / 1000000), axis=-1)
    current_states_valid = data['state/current/valid'] > 0.

    observed_states = np.concatenate((past_states, current_states), axis=1)
    observed_states = np.concatenate((observed_positions, observed_states), axis=-1)
    is_valid_mask = np.concatenate((past_states_valid, current_states_valid), axis=1)
    point_mask = np.logical_and(fov_mask, is_valid_mask) # is_valid_mask

    #any_observed_states_valid_mask = np.sum(point_mask, axis=1) > 0
    #observed_states = observed_states[any_observed_states_valid_mask]
    #point_mask = point_mask[any_observed_states_valid_mask]
    #observed_states = np.where(point_mask[..., None], observed_states, np.nan)

    return torch.FloatTensor(observed_states)

def collate_roadgraph(data):
    roadgraph_points = data['roadgraph_samples/xyz'][:,:2]
    centered_and_rotated_roadgraph_points = normalize_about_sdc(roadgraph_points, data)
    roadgraph_image_points = get_image_coordinates(centered_and_rotated_roadgraph_points)

    fov_mask = get_fov_mask(roadgraph_image_points)
    is_valid_mask = data['roadgraph_samples/valid'] > 0.
    point_mask = np.logical_and(fov_mask.reshape(-1, 1), is_valid_mask)

    point_mask = point_mask.flatten()
    roadgraph_points = roadgraph_image_points[point_mask]
    roadgraph_type = data['roadgraph_samples/type'][point_mask]
    roadgraph_types = np.unique(roadgraph_type)
    roadgraph_id = data['roadgraph_samples/id'][point_mask]

    return roadgraph_points, roadgraph_type, roadgraph_types, roadgraph_id

def collate_traffic_light_state(data):
    traffic_light_points = np.stack((data['traffic_light_state/current/x'][0], data['traffic_light_state/current/y'][0]), axis=-1)
    centered_and_rotated_traffic_light_points = normalize_about_sdc(traffic_light_points, data)
    traffic_light_image_points = get_image_coordinates(centered_and_rotated_traffic_light_points)

    fov_mask = get_fov_mask(traffic_light_image_points)
    is_valid_mask = data['traffic_light_state/current/valid'][0] > 0.
    point_mask = np.logical_and(fov_mask, is_valid_mask)

    traffic_light_image_points = traffic_light_image_points[point_mask]
    traffic_light_state = data['traffic_light_state/current/state'][0][point_mask]

    return traffic_light_image_points, traffic_light_state

def extract_lines(xy, id, typ):
    points = [] 
    lines = []
    length = xy.shape[0]
    for i, p in enumerate(xy):
        points.append(p)
        next_id = id[i+1] if i < length-1 else id[i]
        current_id = id[i]
        if next_id != current_id or i == length-1:
            if typ in [18, 19]:
                points.append(points[0])
            lines.append(points)
            points = []
    return lines

def rasterize_road_map(data):
    roadgraph_points, roadgraph_type, roadgraph_types, roadgraph_id = collate_roadgraph(data)
    traffic_light_points, traffic_light_state = collate_traffic_light_state(data)

    fig, ax = plt.subplots()
    fig.set_size_inches([IMG_SIZE, IMG_SIZE])
    fig.set_dpi(DPI)
    fig.set_tight_layout(True)
    fig.set_facecolor('k')
    ax.set_facecolor('k')
    ax.grid(False)
    ax.margins(0)
    ax.axis('off')

    # plot static roadmap
    big=80
    for t in roadgraph_types:
        road_points = roadgraph_points[np.where(roadgraph_type==t)[0]]
        point_id = roadgraph_id[np.where(roadgraph_type==t)[0]]
        if t in set([1, 2, 3]):
            lines = extract_lines(road_points, point_id, t)
            for line in lines:
                ax.plot([point[0] for point in line], [point[1] for point in line], 
                        color=road_line_map[t][0], linestyle=road_line_map[t][1], linewidth=road_line_map[t][2]*big, alpha=1, zorder=1)
        elif t == 17: # plot stop signs
            ax.plot(road_points.T[0, :], road_points.T[1, :], road_line_map[t][1], color=road_line_map[t][0], markersize=road_line_map[t][2]*big)
        elif t in set([18, 19]): # plot crosswalk and speed bump
            rects = extract_lines(road_points, point_id, t)
            for rect in rects:
                area = plt.fill([point[0] for point in rect], [point[1] for point in rect], color=road_line_map[t][0], alpha=0.7, zorder=2)
        else: # plot other elements
            lines = extract_lines(road_points, point_id, t)
            for line in lines:
                ax.plot([point[0] for point in line], [point[1] for point in line], 
                        color=road_line_map[t][0], linestyle=road_line_map[t][1], linewidth=road_line_map[t][2]*big)

    # plot traffic lights
    for lp, ls in zip(traffic_light_points, traffic_light_state):
        light_circle = plt.Circle(lp, 0.08*big, color=light_state_map[ls], zorder=2)
        ax.add_artist(light_circle)

    ax.axis([0, GRID_SIZE, 0, GRID_SIZE])
    ax.set_aspect('equal')
    fig.canvas.draw()
    road_map = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]

    plt.close('all')

    return torch.FloatTensor(road_map)

def collate_target_flow_field(data):
    type_mask = data['state/type'] == 1
    unobserved_positions = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)
    unobserved_positions = unobserved_positions[type_mask]

    max_agents, timesteps, xy = unobserved_positions.shape
    unobserved_positions = unobserved_positions.reshape(-1, xy)
    
    centered_and_rotated_unobserved_positions = normalize_about_sdc(unobserved_positions, data)
    centered_and_rotated_image_unobserved_positions = get_image_coordinates(centered_and_rotated_unobserved_positions)
    fov_mask = get_fov_mask(centered_and_rotated_image_unobserved_positions)

    unobserved_positions = centered_and_rotated_unobserved_positions.reshape(max_agents, timesteps, xy)
    fov_mask = fov_mask.reshape(max_agents, timesteps)

    future_times = data['state/future/timestamp_micros'] / 1000000
    future_velocity = np.stack((data['state/future/velocity_x'], data['state/future/velocity_y']), axis=-1)

    future_times = future_times[type_mask]
    future_velocity = future_velocity[type_mask]

    is_valid_mask = data['state/future/valid'][type_mask] > 0.
    point_mask = np.logical_and(fov_mask, is_valid_mask)

    unobserved_positions = unobserved_positions.reshape(-1, 2)
    future_times = future_times.reshape(-1)
    future_velocity = future_velocity.reshape(-1, 2)
    point_mask = point_mask.reshape(-1)

    unobserved_positions = unobserved_positions[point_mask]
    future_times = future_times[point_mask]
    future_velocity = future_velocity[point_mask]

    return torch.FloatTensor(unobserved_positions), torch.FloatTensor(future_times), torch.FloatTensor(future_velocity)

def collate_target_occupancy_grid(data):
    future_positions = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)
    # TODO: collate ground truth occupancy grid
    return torch.rand(224, 224, 3) # placeholder

def waymo_collate_fn(batch):
    road_maps = []
    agent_trajectories = []
    unobserved_positions = []
    future_times = []
    future_velocities = []
    target_occupancy_grids = []

    for data in batch:
        # TODO: if we want to batch we need to be able to handle variable sized tensors
        road_maps.append(rasterize_road_map(data))
        agent_trajectories.append(collate_agent_trajectories(data))
        pos, t, vel = collate_target_flow_field(data)
        unobserved_positions.append(pos)
        future_times.append(t)
        future_velocities.append(vel)
        target_occupancy_grids.append(collate_target_occupancy_grid(data))

    road_map_batch = torch.stack(road_maps, dim=0)
    agent_trajectories_batch = torch.stack(agent_trajectories, dim=0)
    unobserved_positions_batch = torch.stack(unobserved_positions, dim=0)
    future_times_batch = torch.stack(future_times, dim=0)
    future_velocities_batch = torch.stack(future_velocities, dim=0)
    target_occupancy_grid_batch = torch.stack(target_occupancy_grids, dim=0)

    return road_map_batch, agent_trajectories_batch, unobserved_positions_batch, future_times_batch, future_velocities_batch, target_occupancy_grid_batch
