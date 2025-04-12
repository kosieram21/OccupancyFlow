import os
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datasets.Waymo import get_image_coordinates, get_image_velocity

def render_observed_scene_state(road_map, agent_trajectories, save_path=None):
    image_buffer = road_map.numpy() / 255.0

    plt.title('Current State (t = 1.0s)')
    plt.imshow(image_buffer)
    #plt.axis('off')

    agent_cmap = ['blue', 'orange', 'yellow', 'purple']

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