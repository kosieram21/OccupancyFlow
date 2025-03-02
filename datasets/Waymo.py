import os
import glob
import numpy as np
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
from subprocess import call
from tqdm import tqdm

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

def transform(feature):
    transform = features_transforms
    keys = transform.keys()
    for key in keys:
        func = transform[key]
        feat = feature[key]
        feature[key] = func(feat)
    return feature

def create_idx(tfrecord_dir, idx_dir):
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

def collate_agent_trajectories(data):
    # TODO: delete these prints
    #print('past times:')
    #print(f" observed: {data['state/past/timestamp_micros'][0] / 1000000}")
    #print(f' theoretical: {[i / 10 for i in range(10)]}')
    #print('current time:')
    #print(f" observed: {data['state/current/timestamp_micros'][0] / 1000000}")
    #print(' theoretical: [1.0]')
    #print('future times:')
    #print(f" observed: {data['state/future/timestamp_micros'][0] / 1000000}")
    #print(f' theoretical: {[i / 10 for i in range(11, 91)]}')

    past_states = np.stack((data['state/past/x'], data['state/past/y'], data['state/past/bbox_yaw'],
                            data['state/past/velocity_x'], data['state/past/velocity_y'], data['state/past/vel_yaw'],
                            data['state/past/width'], data['state/past/length'],
                            data['state/past/timestamp_micros']), axis=-1)
    past_states_valid = data['state/past/valid'] > 0.

    current_states = np.stack((data['state/current/x'], data['state/current/y'], data['state/current/bbox_yaw'],
                               data['state/current/velocity_x'], data['state/current/velocity_y'], data['state/current/vel_yaw'],
                               data['state/current/width'], data['state/current/length'],
                               data['state/current/timestamp_micros']), axis=-1)
    current_states_valid = data['state/current/valid'] > 0.

    observed_states = np.concatenate((past_states, current_states), axis=1)
    observed_states_valid = np.concatenate((past_states_valid, current_states_valid), axis=1)

    any_observed_states_valid_mask = np.sum(observed_states_valid, axis=1) > 0
    observed_states = observed_states[any_observed_states_valid_mask]
    observed_states_valid = observed_states_valid[any_observed_states_valid_mask]
    observed_states = np.where(observed_states_valid[..., None], observed_states, np.nan)

    return torch.FloatTensor(observed_states)

# TODO: delete
def collate_road_graph(data):
    # [20000x6]
    road_graph = np.concatenate((data['roadgraph_samples/id'], 
                                 data['roadgraph_samples/type'], 
                                 data['roadgraph_samples/xyz'][:,:2], 
                                 data['roadgraph_samples/dir'][:,:2]), axis=-1)
    # TODO: when would the road graph not be valid and what should we do for non-valid graph cells
    road_graph_valid = data['roadgraph_samples/valid'] > 0.

    return torch.FloatTensor(road_graph)

# TODO: delete
def collate_traffic_light_state(data):
    # [16x11x3] (what is the 16? number of lights in total?)
    # SceneTransformer transposed these... why? I think so the first dim is per light (if assumtion that 16 = max lights)
    past_traffic_light_states = np.stack((data['traffic_light_state/past/state'].T,
                                          data['traffic_light_state/past/x'].T,
                                          data['traffic_light_state/past/y'].T), axis=-1)
    past_traffic_light_states_valid = data['traffic_light_state/past/valid'].T > 0.
    
    current_traffic_light_states = np.stack((data['traffic_light_state/current/state'].T,
                                             data['traffic_light_state/current/x'].T,
                                             data['traffic_light_state/current/y'].T), axis=-1)
    current_traffic_light_states_valid = data['traffic_light_state/current/valid'].T > 0.

    traffic_light_states = np.concatenate((past_traffic_light_states, current_traffic_light_states), axis=1)
    # TODO: when would the traffic light state not be valid and what should we do for non-valid states
    traffic_light_states_valid = np.concatenate((past_traffic_light_states_valid, current_traffic_light_states_valid), axis=1)

    return torch.FloatTensor(traffic_light_states)

def collate_road_map(data):
    # TODO: build rasterized rgb image from road graph and traffic light states
    road_map = torch.rand(224, 224, 3) # should be 256x256x3
    return road_map

def collate_target_flow_field(data):
    # TODO: collate ground truth flow field
    future_positions = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)
    future_velocities = np.stack((data['state/future/velocity_x'], data['state/future/velocity_y']), axis=-1)
    future_states_valid = data['state/future/valid'] > 0.

    #print(future_positions.shape)
    #print(future_velocities.shape)
    #print(future_states_valid.shape)

    # filter future_positions and future_velocities by future_states_valid
    # produce flow estimate tensor

    return torch.rand(224, 224, 3) # placeholder

def collate_target_occupancy_grid(data):
    future_positions = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)
    # TODO: collate ground truth occupancy grid
    return torch.rand(224, 224, 3) # placeholder

def waymo_collate_fn(batch):
    road_maps = []
    agent_trajectories = []
    target_flow_fields = []
    target_occupancy_grids = []

    for data in batch:
        road_maps.append(collate_road_map(data))
        agent_trajectories.append(collate_agent_trajectories(data))
        target_flow_fields.append(collate_target_flow_field(data))
        target_occupancy_grids.append(collate_target_occupancy_grid(data))

    road_map_batch = torch.stack(road_maps, dim=0)
    agent_trajectories_batch = torch.stack(agent_trajectories, dim=0)
    target_flow_field_batch = torch.stack(target_flow_fields, dim=0)
    target_occupancy_grid_batch = torch.stack(target_occupancy_grids, dim=0)

    return road_map_batch, agent_trajectories_batch, target_flow_field_batch, target_occupancy_grid_batch