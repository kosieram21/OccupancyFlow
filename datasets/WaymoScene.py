import torch
from dataclasses import dataclass

@dataclass
class ObservedState:
    road_map: torch.Tensor
    agent_trajectories: torch.Tensor
    agent_mask: torch.Tensor

@dataclass
class FlowField:
    positions: torch.Tensor
    times: torch.Tensor
    velocities: torch.Tensor
    agent_ids: torch.Tensor
    flow_mask: torch.Tensor

@dataclass
class OccupancyGrid:
    positions: torch.Tensor
    times: torch.Tensor
    unoccluded_occupancies: torch.Tensor
    occluded_occupancies: torch.Tensor

@dataclass
class WaymoScene:
    observed_state: ObservedState
    flow_field: FlowField
    occupancy_grid: OccupancyGrid