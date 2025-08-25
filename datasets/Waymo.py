import os
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import torch
import torch.nn.functional as F
from tfrecord.torch.dataset import MultiTFRecordDataset
from subprocess import call
from tqdm import tqdm
from WaymoScene import WaymoScene, ObservedState, FlowField, OccupancyGrid

PIXELS_PER_METER = 3.2
SDC_X_IN_GRID = 128
SDC_Y_IN_GRID = 128

GRID_SIZE = 256
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
    if tfrecord_dir == idx_dir:
        raise ValueError(f"tfrecord_dir is the same as idx_dir, aborting operation to avoid data corruption")

    os.makedirs(idx_dir, exist_ok=True)

    for tfrecord in tqdm(glob.glob(tfrecord_dir+'/*')):
        idxname = idx_dir + '/' + tfrecord.split('/')[-1]
        call(["tfrecord2idx", tfrecord, idxname])

def WaymoDataset(tfrecord_dir, idx_dir, fold=0, num_folds=1):
    tfrecord_pattern = tfrecord_dir+'/{}'
    index_pattern = idx_dir+'/{}'
    fnlist = os.listdir(tfrecord_pattern.split('{}')[0])
    num_records = len(fnlist)
    fold_size = num_records // num_folds
    remainder = num_records % num_folds
    start = fold_size * fold + min(fold, remainder)
    end = fold_size * (fold + 1) + min(fold + 1, remainder)
    splits = {fn: 1/len(fnlist) for fn in fnlist[start:end]}
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

    return centered_and_rotated_points, angle, sdc_xy

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
    
    return points, angle, sdc_xy

def get_image_coordinates(world_points):
    scale = np.array([PIXELS_PER_METER, -PIXELS_PER_METER])
    offset = np.array([SDC_X_IN_GRID, SDC_Y_IN_GRID])
    image_points = world_points * scale + offset
    return image_points

def get_world_coordinates(image_points):
    offset = np.array([SDC_X_IN_GRID, SDC_Y_IN_GRID])
    scale = np.array([PIXELS_PER_METER, -PIXELS_PER_METER])
    world_points = (image_points - offset) / scale
    return world_points

def get_image_velocity(world_velocity):
    scale = np.array([PIXELS_PER_METER, -PIXELS_PER_METER])
    image_velocity = world_velocity * scale
    return image_velocity

def get_world_velocity(image_velocity):
    scale = np.array([PIXELS_PER_METER, -PIXELS_PER_METER])
    world_velocity = image_velocity / scale
    return world_velocity

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

    max_agents, timesteps, xy = observed_positions.shape
    observed_positions = observed_positions.reshape(-1, xy)
    
    centered_and_rotated_observed_positions, angle, translation = normalize_about_sdc(observed_positions, data)
    centered_and_rotated_observed_positions[:, 1] = -centered_and_rotated_observed_positions[:, 1]
    
    centered_and_rotated_image_observed_positions = get_image_coordinates(centered_and_rotated_observed_positions)
    fov_mask = get_fov_mask(centered_and_rotated_image_observed_positions)

    observed_positions = centered_and_rotated_observed_positions.reshape(max_agents, timesteps, xy)
    fov_mask = fov_mask.reshape(max_agents, timesteps)

    past_velocity = np.stack((data['state/past/velocity_x'], data['state/past/velocity_y']), axis=-1)
    past_velocity = rotate_points_around_origin(past_velocity, angle)
    past_bbox_yaw = -data['state/past/bbox_yaw'] - angle
    past_vel_yaw = -data['state/past/vel_yaw'] - angle
    past_states = np.stack((
        past_bbox_yaw, 
        data['state/past/width'], 
        data['state/past/length'],
        past_vel_yaw, 
        past_velocity[:, :, 0],
        -past_velocity[:, :, 1],
        data['state/past/timestamp_micros'] / 1000000
    ), axis=-1)
    past_states_valid = data['state/past/valid'] > 0.

    current_velocity = np.stack((data['state/current/velocity_x'], data['state/current/velocity_y']), axis=-1)
    current_velocity = rotate_points_around_origin(current_velocity, angle)
    current_bbox_yaw = -data['state/current/bbox_yaw'] - angle
    current_vel_yaw = -data['state/current/vel_yaw'] - angle
    current_states = np.stack((
        current_bbox_yaw, 
        data['state/current/width'],
        data['state/current/length'],
        current_vel_yaw, 
        current_velocity[:, :, 0],
        -current_velocity[:, :, 1],
        data['state/current/timestamp_micros'] / 1000000
    ), axis=-1)
    current_states_valid = data['state/current/valid'] > 0.

    observed_states = np.concatenate((past_states, current_states), axis=1)
    observed_states = np.concatenate((observed_positions, observed_states), axis=-1)

    agent_type = np.reshape(data['state/type'], (max_agents, 1, 1))
    agent_type = np.tile(agent_type, (1, timesteps, 1))
    observed_states = np.concatenate((observed_states, agent_type), axis=-1)
    
    is_valid_mask = np.concatenate((past_states_valid, current_states_valid), axis=1)
    point_mask = np.logical_and(fov_mask, is_valid_mask)

    any_observed_states_valid_mask = np.sum(point_mask, axis=1) > 0
    observed_states = observed_states[any_observed_states_valid_mask]
    point_mask = point_mask[any_observed_states_valid_mask]
    observed_states = np.where(point_mask[..., None], observed_states, np.nan)

    return torch.FloatTensor(observed_states)

def collate_roadgraph(data):
    roadgraph_points = data['roadgraph_samples/xyz'][:,:2]
    centered_and_rotated_roadgraph_points, angle, translation = normalize_about_sdc(roadgraph_points, data)
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
    centered_and_rotated_traffic_light_points, angle, translation = normalize_about_sdc(traffic_light_points, data)
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

def expand_to_bounding_box(positions, lengths, widths, values = None, step_size=1.0):
    expanded = []

    for i in range(positions.shape[0]):
        x_center, y_center = positions[i]
        width = widths[i][0]
        length = lengths[i][0]
        
        x_min = x_center - length / 2
        x_max = x_center + length / 2
        y_min = y_center - width / 2
        y_max = y_center + width / 2
        
        x_pos = np.arange(x_min, x_max, step_size)
        y_pos = np.arange(y_min, y_max, step_size)

        grid_x, grid_y = np.meshgrid(x_pos, y_pos)

        if values is None:
            expanded.extend(np.c_[grid_x.ravel(), grid_y.ravel()])
        else:
            value = values[i]
            value_grid = []
            for j in range(value.shape[0]):
                value_grid.append(np.full_like(grid_x, value[j]))
            expanded.extend(np.column_stack([v.ravel() for v in value_grid]))

    return np.array(expanded)

def collate_target_flow_fields(data):
    vehicle_type_mask = data['state/type'] == 1
    vehicle_agent_ids, vehicle_agent_positions, vehicle_agent_times, vehicle_agent_velocities = collate_target_flow_field(data, vehicle_type_mask)

    #pedestrian_type_mask = data['state/type'] == 2
    #pedestrian_agent_ids, pedestrian_agent_positions, pedestrian_agent_times, pedestrian_agent_velocities = collate_target_flow_field(data, pedestrian_type_mask)

    #cyclist_type_mask = data['state/type'] == 3
    #cyclist_agent_ids, cyclist_agent_positions, cyclist_agent_times, cyclist_agent_velocities = collate_target_flow_field(data, pedestrian_type_mask)

    return vehicle_agent_ids, vehicle_agent_positions, vehicle_agent_times, vehicle_agent_velocities

def collate_target_flow_field(data, type_mask):
    if not type_mask.any():
        agent_ids = torch.zeros((1, 1))
        agent_positions = torch.zeros((1, 2))
        agent_times = torch.zeros((1, 1))
        agent_velocities = torch.zeros((1, 2))
        return (agent_ids, agent_positions, agent_times, agent_velocities)

    past_positions = np.stack((data['state/past/x'], data['state/past/y']), axis=-1)
    current_position = np.stack((data['state/current/x'], data['state/current/y']), axis=-1)
    future_positions = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)
    agent_positions = np.concatenate((past_positions, current_position, future_positions), axis=1)
    agent_positions = agent_positions[type_mask]

    max_agents, timesteps, xy = agent_positions.shape
    agent_positions = agent_positions.reshape(-1, xy)

    centered_and_rotated_agent_positions, angle, translation = normalize_about_sdc(agent_positions, data)
    centered_and_rotated_agent_positions[:, 1] = -centered_and_rotated_agent_positions[:, 1]
    centered_and_rotated_image_agent_positions = get_image_coordinates(centered_and_rotated_agent_positions)

    fov_mask = get_fov_mask(centered_and_rotated_image_agent_positions)
    fov_mask = fov_mask.reshape(max_agents, timesteps)

    is_valid = np.concatenate((data['state/past/valid'], data['state/current/valid'], data['state/future/valid']), axis=1)
    is_valid_mask = is_valid[type_mask] > 0.
    point_mask = np.logical_and(fov_mask, is_valid_mask)
    point_mask = point_mask.reshape(-1)

    agent_lengths = np.concatenate((data['state/past/length'], data['state/current/length'], data['state/future/length']), axis=1)
    agent_lengths = agent_lengths[type_mask]
    max_length = np.max(agent_lengths, axis=1, keepdims=True)
    agent_lengths = np.repeat(max_length, timesteps, axis=1).reshape(-1, 1)
    agent_lengths = agent_lengths[point_mask]
    
    agent_widths = np.concatenate((data['state/past/width'], data['state/current/width'], data['state/future/width']), axis=1)
    agent_widths = agent_widths[type_mask]
    max_width = np.max(agent_widths, axis=1, keepdims=True)
    agent_widths = np.repeat(max_width, timesteps, axis=1).reshape(-1, 1)
    agent_widths = agent_widths[point_mask]

    agent_bbox_yaws = np.concatenate((data['state/past/bbox_yaw'], data['state/current/bbox_yaw'], data['state/future/bbox_yaw']), axis=1)
    agent_bbox_yaws = agent_bbox_yaws[type_mask].reshape(-1, 1)
    agent_bbox_yaws = agent_bbox_yaws[point_mask]

    agent_ids = data['state/id'][type_mask]
    agent_ids = np.repeat(agent_ids[:, np.newaxis], timesteps, axis=1)
    agent_ids = agent_ids.reshape(-1, 1)
    agent_ids = agent_ids[point_mask]

    agent_positions = centered_and_rotated_agent_positions.reshape(max_agents, timesteps, xy)
    agent_positions = agent_positions.reshape(-1, 2)
    agent_positions = agent_positions[point_mask]

    agent_times = np.concatenate((data['state/past/timestamp_micros'], data['state/current/timestamp_micros'], data['state/future/timestamp_micros']), axis=1)
    agent_times = agent_times / 1000000
    agent_times = agent_times[type_mask]
    agent_times = agent_times.reshape(-1, 1)
    agent_times = agent_times[point_mask]

    past_velocities = np.stack((data['state/past/velocity_x'], data['state/past/velocity_y']), axis=-1)
    current_velocity = np.stack((data['state/current/velocity_x'], data['state/current/velocity_y']), axis=-1)
    future_velocities = np.stack((data['state/future/velocity_x'], data['state/future/velocity_y']), axis=-1)
    agent_velocities = np.concatenate((past_velocities, current_velocity, future_velocities), axis=1)
    agent_velocities = rotate_points_around_origin(agent_velocities, angle)
    agent_velocities[:, :, 1] = -agent_velocities[:, :, 1]
    agent_velocities = agent_velocities[type_mask]
    agent_velocities = agent_velocities.reshape(-1, 2)
    agent_velocities = agent_velocities[point_mask]

    agent_ids = expand_to_bounding_box(agent_positions, agent_lengths, agent_widths, agent_ids)
    agent_centers = expand_to_bounding_box(agent_positions, agent_lengths, agent_widths, agent_positions)
    agent_times = expand_to_bounding_box(agent_positions, agent_lengths, agent_widths, agent_times)
    agent_velocities = expand_to_bounding_box(agent_positions, agent_lengths, agent_widths, agent_velocities)
    agent_bbox_yaws = expand_to_bounding_box(agent_positions, agent_lengths, agent_widths, agent_bbox_yaws)
    agent_positions = expand_to_bounding_box(agent_positions, agent_lengths, agent_widths)

    agent_positions = agent_positions - agent_centers
    for i in range(agent_positions.shape[0]):
        agent_positions[i] = rotate_points_around_origin(agent_positions[i], -agent_bbox_yaws[i] - angle)
    agent_positions = agent_positions + agent_centers

    return (
        torch.FloatTensor(agent_ids), 
        torch.FloatTensor(agent_positions), 
        torch.FloatTensor(agent_times), 
        torch.FloatTensor(agent_velocities),
    )

def points_contained_in_bbox(points, center, length, width, yaw):
    x_min = -length / 2
    x_max = length / 2
    y_min = -width / 2
    y_max = width / 2

    corners = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min]
    ])

    corners = rotate_points_around_origin(corners, yaw)
    corners = corners + center

    bbox = Path(corners)
    
    mask = bbox.contains_points(points.reshape(-1, 2).numpy())
    mask = torch.as_tensor(mask.reshape(points.shape[:-1]), dtype=torch.bool)

    return mask

def collate_target_occupancy_grids(data):
    type_mask = data['state/type'] == 1 # vehicles... TODO: we should take this in as an argument

    observed_agents = set()

    past_positions = np.stack((data['state/past/x'], data['state/past/y']), axis=-1)
    current_position = np.stack((data['state/current/x'], data['state/current/y']), axis=-1)
    future_positions = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)
    agent_positions = np.concatenate((past_positions, current_position, future_positions), axis=1)
    agent_positions = agent_positions[type_mask]

    max_agents, timesteps, xy = agent_positions.shape

    agent_lengths = np.concatenate((data['state/past/length'], data['state/current/length'], data['state/future/length']), axis=1)
    agent_lengths = agent_lengths[type_mask]
    
    agent_widths = np.concatenate((data['state/past/width'], data['state/current/width'], data['state/future/width']), axis=1)
    agent_widths = agent_widths[type_mask]

    agent_bbox_yaws = np.concatenate((data['state/past/bbox_yaw'], data['state/current/bbox_yaw'], data['state/future/bbox_yaw']), axis=1)
    agent_bbox_yaws = agent_bbox_yaws[type_mask]

    agent_ids = data['state/id'][type_mask]

    is_valid = np.concatenate((data['state/past/valid'], data['state/current/valid'], data['state/future/valid']), axis=1)
    is_valid_mask = is_valid[type_mask] > 0.

    # TODO: can we make this a shared reusable function?
    y_coords = np.arange(0, GRID_SIZE, 1)
    x_coords = np.arange(0, GRID_SIZE, 1)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    grid_points = np.stack((grid_x, grid_y), axis=-1)
    grid_points = get_world_coordinates(grid_points)
    grid_points = torch.FloatTensor(grid_points)
    #grid_points = grid_points.unsqueeze(2).repeat(1, 1, timesteps, 1)

    grid_times = torch.arange(timesteps, dtype=torch.float32) / 10
    #grid_times = grid_times.view(1, 1, timesteps, 1).repeat(GRID_SIZE, GRID_SIZE, 1, 1)
    # END TODO

    occupancy_grid = torch.zeros_like(grid_times)
    occupancy_grid = occupancy_grid.view(1, 1, timesteps, 1).repeat(GRID_SIZE, GRID_SIZE, 1, 1)

    occluded_occupancy_grid = torch.zeros_like(grid_times)
    occluded_occupancy_grid = occluded_occupancy_grid.view(1, 1, timesteps, 1).repeat(GRID_SIZE, GRID_SIZE, 1, 1)

    for t in range(timesteps):
        grid_points_at_time = grid_points
        occupancy_grid_at_time = occupancy_grid[:, :, t, :]
        occluded_occupancy_grid_at_time = occluded_occupancy_grid[:, :, t, :]

        agent_positions_at_time = agent_positions[:, t]
        agent_lengths_at_time = agent_lengths[:, t]
        agent_widths_at_time = agent_widths[:, t]
        agent_bbox_yaws_at_time = agent_bbox_yaws[:, t]
        is_valid_mask_at_time = is_valid_mask[:, t]

        centered_and_rotated_agent_positions_at_time, angle, translation = normalize_about_sdc(agent_positions_at_time, data)
        centered_and_rotated_agent_positions_at_time[:, 1] = -centered_and_rotated_agent_positions_at_time[:, 1]
        centered_and_rotated_image_agent_positions_at_time = get_image_coordinates(centered_and_rotated_agent_positions_at_time)

        fov_mask_at_time = get_fov_mask(centered_and_rotated_image_agent_positions_at_time)
        point_mask_at_time = np.logical_and(fov_mask_at_time, is_valid_mask_at_time)

        centered_and_rotated_agent_positions_at_time = centered_and_rotated_agent_positions_at_time[point_mask_at_time]
        agent_lengths_at_time = agent_lengths_at_time[point_mask_at_time]
        agent_widths_at_time = agent_widths_at_time[point_mask_at_time]
        agent_bbox_yaws_at_time = agent_bbox_yaws_at_time[point_mask_at_time]
        agent_ids_at_time = agent_ids[point_mask_at_time]

        for i in range(len(agent_ids_at_time)):
            center = centered_and_rotated_agent_positions_at_time[i]
            length = agent_lengths_at_time[i]
            width = agent_widths_at_time[i]
            yaw = agent_bbox_yaws_at_time[i]
            id = agent_ids_at_time[i]

            bbox_mask = points_contained_in_bbox(grid_points_at_time, center, length, width, -yaw - angle)

            if t < 11 or id in observed_agents:
                occupancy_grid_at_time[bbox_mask] = 1.0
                observed_agents.add(id)
            else:
                occluded_occupancy_grid_at_time[bbox_mask] = 1.0

    return grid_points, grid_times, occupancy_grid, occluded_occupancy_grid

def pad_tensors(tensors, max_size):
    padded_tensors = []
    masks = []

    for tensor in tensors:
        samples = tensor.shape[0]
        padding_size = max_size - samples
        
        padding = [0] * (2 * tensor.dim())
        padding[-1] = padding_size
        padded_tensor = F.pad(tensor, padding)
        padded_tensors.append(padded_tensor)
        
        mask = torch.ones(max_size, dtype=torch.bool)
        mask[samples:] = 0
        masks.append(mask)

    return padded_tensors, masks

def waymo_collate_fn(batch):
    road_maps = []
    agent_trajectories = []

    flow_field_agent_ids = []
    flow_field_positions = []
    flow_field_times = []
    flow_field_velocities = []

    occupancy_grid_positions = []
    occupancy_grid_times = []
    occupancy_grid_unoccluded_occupancies = []
    occupancy_grid_occluded_occupancies = []

    vehicle_flow_field_agent_ids = []
    vehicle_flow_field_positions = []
    vehicle_flow_field_times = []
    vehicle_flow_field_velocities = []

    pedestrian_flow_field_agent_ids = []
    pedestrian_flow_field_positions = []
    pedestrian_flow_field_times = []
    pedestrian_flow_field_velocities = []

    cyclist_flow_field_agent_ids = []
    cyclist_flow_field_positions = []
    cyclist_flow_field_times = []
    cyclist_flow_field_velocities = []

    for data in batch:
        # model inputs
        road_maps.append(rasterize_road_map(data))
        agent_trajectories.append(collate_agent_trajectories(data))

        # ground truth flow field
        ids, pos, t, vel = collate_target_flow_fields(data)
        flow_field_agent_ids.append(ids)
        flow_field_positions.append(pos)
        flow_field_times.append(t)
        flow_field_velocities.append(vel)

        # TODO: do we need something for occupancy alignment finetuning?

        # ground truth occupancy gird
        grid_points, grid_times, occupancy_grid, occluded_occupancy_grid = collate_target_occupancy_grids(data)
        occupancy_grid_positions.append(grid_points)
        occupancy_grid_times.append(grid_times)
        occupancy_grid_unoccluded_occupancies.append(occupancy_grid)
        occupancy_grid_occluded_occupancies.append(occluded_occupancy_grid)

    max_agents = max(t.shape[0] for t in agent_trajectories)
    max_agent_positions = max(p.shape[0] for p in flow_field_positions)
    agent_trajectories, agent_mask = pad_tensors(agent_trajectories, max_agents)
    flow_field_agent_ids, flow_field_mask = pad_tensors(flow_field_agent_ids, max_agent_positions)
    flow_field_positions, _ = pad_tensors(flow_field_positions, max_agent_positions)
    flow_field_times, _ = pad_tensors(flow_field_times, max_agent_positions)
    flow_field_velocities, _ = pad_tensors(flow_field_velocities, max_agent_positions)

    road_map_batch = torch.stack(road_maps, dim=0)
    agent_trajectories_batch = torch.stack(agent_trajectories, dim=0)

    flow_field_positions_batch = torch.stack(flow_field_positions, dim=0)
    flow_field_times_batch = torch.stack(flow_field_times, dim=0)
    flow_field_velocities_batch = torch.stack(flow_field_velocities, dim=0)
    flow_field_agent_ids_batch = torch.stack(flow_field_agent_ids, dim=0)
    
    occupancy_grid_positions_batch = torch.stack(occupancy_grid_positions, dim=0)
    occupancy_grid_times_batch = torch.stack(occupancy_grid_times, dim=0)
    occupancy_grid_unoccluded_occupancies_batch = torch.stack(occupancy_grid_unoccluded_occupancies, dim=0)
    occupancy_grid_occluded_occupancies_batch = torch.stack(occupancy_grid_occluded_occupancies, dim=0)

    agent_mask_batch = torch.stack(agent_mask, dim=0)
    flow_field_mask_batch = torch.stack(flow_field_mask, dim=0)

    # TODO: ODE fintuning data should be pre-collated as well
    waymo_scene = WaymoScene.from_tensors(
        road_map_batch, agent_trajectories_batch,
        flow_field_positions_batch, flow_field_times_batch, flow_field_velocities_batch, flow_field_agent_ids_batch,
        occupancy_grid_positions_batch, occupancy_grid_times_batch, occupancy_grid_unoccluded_occupancies_batch, occupancy_grid_occluded_occupancies_batch,
        agent_mask_batch, flow_field_mask_batch
    )

    return waymo_scene
