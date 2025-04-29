import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datasets.Waymo import get_image_coordinates, get_image_velocity, rotate_points_around_origin

agent_cmap = ['blue', 'orange', 'yellow', 'purple']

def render_observed_scene_state_current_timestep(road_map, agent_trajectories, save_path=None):
    image_buffer = road_map.numpy() / 255.0

    fig, ax = plt.subplots()
    ax.imshow(image_buffer)
    ax.axis('off')

    for agent in range(agent_trajectories.shape[0]):
        agent_x = agent_trajectories[agent, -1, 0].item()
        agent_y = agent_trajectories[agent, -1, 1].item()
        agent_bbox_yaw = agent_trajectories[agent, -1, 2].item()
        agent_width = agent_trajectories[agent, -1, 3].item()
        agent_length = agent_trajectories[agent, -1, 4].item()
        agent_type = agent_trajectories[agent,-1, -1].item()
        agent_type = 4 if math.isnan(agent_type) else int(agent_type)
        agent_color = agent_cmap[agent_type - 1]
    
        half_w = agent_width / 2
        half_l = agent_length / 2
        corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
        corners = rotate_points_around_origin(corners, agent_bbox_yaw)
        corners += np.array([agent_x, agent_y])
        corners = get_image_coordinates(corners)

        ax.plot(
            [*corners[:, 0], corners[0, 0]],
            [*corners[:, 1], corners[0, 1]],
            color=agent_color, linewidth=2
        )

        heading = np.array([[0.75 * agent_length, 0]])
        heading = rotate_points_around_origin(heading, -agent_bbox_yaw)
        heading = get_image_velocity(heading)[0]
        center = get_image_coordinates(np.array([[agent_x, agent_y]]))[0]
        ax.arrow(center[0], center[1],
                 heading[0], heading[1],
                 head_width=1.0, head_length=1.5,
                 fc='orange', ec='orange')

    ax.set_xlim(0, image_buffer.shape[1])
    ax.set_ylim(image_buffer.shape[0], 0)
        
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def render_observed_scene_state(road_map, agent_trajectories, save_path=None):
    image_buffer = road_map.numpy() / 255.0

    plt.title('Current State (t = 1.0s)')
    plt.imshow(image_buffer)
    plt.axis('off')

    trajectories = get_image_coordinates(agent_trajectories[:,:,:2])
    for agent in range(trajectories.shape[0]):
        agent_trajectory = trajectories[agent, :, :]
        agent_type = agent_trajectories[agent,-1,-1].item()
        agent_type = 4 if math.isnan(agent_type) else int(agent_type)
        agent_color = agent_cmap[agent_type - 1]
        plt.plot(agent_trajectory[:, 0], agent_trajectory[:, 1], marker='o', markersize=3, color=agent_color)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def group_indicies(tensor):
    groups = defaultdict(list)
    tensor = torch.round(tensor * 10)
    for idx, val in enumerate(tensor):
        val = val.item() / 10
        groups[val].append(idx)
    return groups

def render_flow_field(road_map, times, positions, velocity, save_path=None):
    image_buffer = road_map.numpy() / 255.0
    grid_size = image_buffer.shape[0]

    groups = group_indicies(times)
    sorted_keys = sorted(groups.keys())

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.imshow(image_buffer)
        ax.axis('off')

        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)

        time = list(sorted_keys)[frame]
        indices = groups[time]
    
        group_positions = get_image_coordinates(positions[indices])
        group_velocity = get_image_velocity(velocity[indices])

        ax.set_title(f"Future Flow (t = {time}s)")
    
        x_coords, y_coords = zip(*group_positions)
        ax.scatter(x_coords, y_coords, marker='o', s=5, color='blue')
        ax.quiver(x_coords, y_coords, group_velocity[:, 0], group_velocity[:, 1], 
                  angles='xy', scale_units='xy', scale=4.0, 
                  color='orange', width=0.007, 
                  headwidth=4, headlength=5, headaxislength=3)

    anim = FuncAnimation(fig, update, frames=len(groups), repeat=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, fps=10, dpi=300)

    plt.close(fig)

    return anim