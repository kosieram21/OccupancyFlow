from torch.utils.data import DataLoader
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model.Encoder import Encoder

should_index = False

tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
idx_path = '../data/idxs_validation_bs_1'

if should_index:
    create_idx(tfrecord_path, idx_path)

dataset = WaymoDataset(tfrecord_path, idx_path)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: waymo_collate_fn(x))

encoder = Encoder(trajectory_feature_dim=9, 
                  token_dim=128, embedding_dim=1024,
                  motion_encoder_hidden_dim=512, motion_encoder_seq_len=11)

agent_trajectories, road_graph, traffic_light_state, target_flow_field, target_occupancy_grid = next(iter(dataloader))
embedding = encoder(agent_trajectories, road_graph, traffic_light_state)
#for agent_trajectories, road_graph, traffic_light_state, target_flow_field, target_occupancy_grid in dataloader:
#    embedding = encoder(agent_trajectories, road_graph, traffic_light_state)