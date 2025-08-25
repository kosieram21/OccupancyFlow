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

    @classmethod
    def from_tensors(cls,
                     road_map, agent_trajectories,
                     flow_field_positions, flow_field_times, flow_field_velocities, flow_field_agent_ids,
                     occupancy_grid_positions, occupancy_grid_times, occupancy_grid_unoccluded_occupancies, occupancy_grid_occluded_occupancies,
                     agent_mask, flow_mask):
        waymo_scene = cls(
            observed_state=ObservedState(
                road_map=road_map, 
                agent_trajectories=agent_trajectories, 
                agent_mask=agent_mask
            ),
            flow_field=FlowField(
                positions=flow_field_positions, 
                times=flow_field_times, 
                velocities=flow_field_velocities, 
                agent_ids=flow_field_agent_ids, 
                flow_mask=flow_mask
            ),
            occupancy_grid=OccupancyGrid(
                positions=occupancy_grid_positions, 
                times=occupancy_grid_times, 
                unoccluded_occupancies=occupancy_grid_unoccluded_occupancies, 
                occluded_occupancies=occupancy_grid_occluded_occupancies
            ),
        )

        return waymo_scene