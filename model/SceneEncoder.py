import torch
import torch.nn as nn
from model.layers import CDE
from model.layers.CDE import GRUWithZeroFill # DELETE ME
from model.layers import GRU
from model.layers import SelfAttentionTransformer
from model.layers import CrossAttentionTransformer
from model.layers import SwinTransformer

class SceneEncoder(nn.Module):
    def __init__(self, 
                 road_map_image_size, trajectory_feature_dim, 
                 motion_encoder_hidden_dim, motion_encoder_seq_len,
                 visual_encoder_hidden_dim, visual_encoder_window_size,
                 token_dim, embedding_dim):
        super(SceneEncoder, self).__init__()

        assert road_map_image_size % visual_encoder_window_size == 0, "road_map_image_size must be divisible by visual_encoder_window_size"
        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2 for bidirectional GRU"
        assert token_dim % 2 == 0, "token_dim must be divisible by 2 for bidirectional GRU"
        assert token_dim % 8 == 0, "tokken_dim must be divisible by 8 for swin-t patch-merging"

        self.motion_encoder_seq_len = motion_encoder_seq_len
        # self.motion_encoder = CDE(input_dim=trajectory_feature_dim, 
        #                           embedding_dim=token_dim, 
        #                           hidden_dim=motion_encoder_hidden_dim, 
        #                           num_layers=4)
        self.motion_encoder = GRUWithZeroFill(input_dim=trajectory_feature_dim,
                                              hidden_dim=token_dim // 2)
        
        self.visual_encoder = SwinTransformer(img_size=road_map_image_size,
                                              embed_dim=token_dim // 8,#visual_encoder_hidden_dim,
                                              window_size=visual_encoder_window_size,
                                              num_heads=[8,8,8,8])
        
        self.interaction_transformer1 = SelfAttentionTransformer(token_dim=token_dim,
                                                                 num_layers=4,
                                                                 num_heads=8)
        
        self.fusion_transformer = CrossAttentionTransformer(token_dim=token_dim,
                                                            num_layers=4,
                                                            num_heads=8)
        
        self.interaction_transformer2 = SelfAttentionTransformer(token_dim=token_dim,
                                                                 num_layers=4,
                                                                 num_heads=8)
        
        self.pooling_module = GRU(input_dim=token_dim,
                                  hidden_dim=embedding_dim // 2,
                                  num_layers=4,
                                  bidirectional=True)

    def forward(self, road_map, agent_trajectories, agent_mask=None):
        #t = torch.linspace(0., 1., self.motion_encoder_seq_len).to(agent_trajectories)
        #agent_tokens = self.motion_encoder(t, agent_trajectories, agent_mask)
        agent_tokens = self.motion_encoder(agent_trajectories, agent_mask)
        environment_tokens = self.visual_encoder(road_map)
        agent_tokens = self.interaction_transformer1(agent_tokens, agent_mask)
        agent_tokens = agent_tokens + self.fusion_transformer(agent_tokens, environment_tokens, agent_mask)
        agent_tokens = self.interaction_transformer2(agent_tokens, agent_mask)
        embedding = self.pooling_module(agent_tokens, agent_mask)
        return embedding