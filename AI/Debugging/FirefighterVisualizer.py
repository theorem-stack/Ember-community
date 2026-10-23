import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from typing import List, Tuple, Dict, Optional, Deque
from collections import deque
from skimage.transform import resize 

class FirefighterVisualizer:
    """
    Handles 2D top-down visualization for the FirefighterEnvironment.
    Captures frame data and generates animated replays.
    """

    def __init__(self, grid_size: Tuple[int, int], num_agents: int,
                 burn_box_mask: Optional[np.ndarray],
                 sim_dx: float, sim_dy: float, action_type_map: Dict[str, int]):
        
        self.grid_size_x, self.grid_size_y = grid_size
        self.num_agents = num_agents
        
        if burn_box_mask is not None and burn_box_mask.shape == (self.grid_size_x, self.grid_size_y):
            self.burn_box_mask = burn_box_mask.T
        else:
            self.burn_box_mask = burn_box_mask

        self.sim_dx = sim_dx
        self.sim_dy = sim_dy
        self.action_type_map = action_type_map
        
        self._action_id_to_name_map = {v: k for k, v in action_type_map.items()}

        # Fire visualiztion normalization
        self.global_max_intensity = 0.0

        self.target_type_colors = {
            'water': 'deepskyblue',
            'firebreak': 'saddlebrown',
            'retardant': 'red',
            'air_support': 'purple',
            'idle': 'gray',
            'UnknownAction': 'magenta'
        }

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.ax_log: Optional[plt.Axes] = None 
        self.im_fire: Optional[plt.Artist] = None
        self.agent_patches: List[Circle] = []
        self.target_scatter: Optional[plt.Artist] = None
        # CHANGED: Now a list of path lines, one for each agent
        self.path_lines: List[plt.Artist] = [] 
        self.burn_box_artists: List[plt.Artist] = []
        self.time_text: Optional[plt.Artist] = None
        self.step_text: Optional[plt.Artist] = None
        self.action_log_text: Optional[plt.Artist] = None
        self.target_legend_artists: List[plt.Artist] = []

        self.visualization_history: List[Dict] = []
        self._resize_id: Optional[int] = None

        self.action_log: Deque[str] = deque(maxlen=10) 
        self._previous_agent_targets: List[Optional[Dict]] = [None] * self.num_agents

    def _init_plot_artists(self):
        """
        Initializes or re-initializes the matplotlib figure, axes, and all plot artists.
        Ensures both main plot and action log subplot are centered on the x-axis.
        """
        if self.fig:
            plt.close(self.fig)

        # Create a figure with two subplots: main plot on top, log on bottom
        self.fig, (self.ax, self.ax_log) = plt.subplots(2, 1, figsize=(10, 14),
                                                        gridspec_kw={'height_ratios': [3, 1]})

        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_xlabel('X (grid cells)', fontsize=12)
        self.ax.set_ylabel('Y (grid cells)', fontsize=12)

        self.ax.set_xlim(-0.5, self.grid_size_x + 0.5)
        self.ax.set_ylim(-0.5, self.grid_size_y + 0.5)

        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_xticks(np.arange(0, self.grid_size_x + 1, 5))
        self.ax.set_yticks(np.arange(0, self.grid_size_y + 1, 5))

        imshow_extent = [-0.5, self.grid_size_x + 0.5, -0.5, self.grid_size_y + 0.5]

        self.im_fire = self.ax.imshow(np.zeros((self.grid_size_y, self.grid_size_x)), origin='lower',
                              cmap='inferno', vmin=0, vmax=2, alpha=0.7, interpolation='nearest',
                              extent=imshow_extent)

        cbar = self.fig.colorbar(self.im_fire, ax=self.ax, label='Normalized Fire Intensity',
                                orientation='vertical', fraction=0.046, pad=0.10,
                                location='left', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

        colors = ['blue', 'cyan', 'green', 'orange', 'purple', 'brown']
        self.agent_patches = []
        self.path_lines = [] # Initialize list for paths
        for i in range(self.num_agents):
            agent_color = colors[i % len(colors)]
            
            # Agent patch
            patch = Circle((0.5, 0.5), radius=0.4, color=agent_color, zorder=5,
                        label=f'Agent {i}', alpha=0.9)
            self.agent_patches.append(patch)
            self.ax.add_patch(patch)

            # CHANGED: Create a separate path line for each agent
            # Use a slightly different style or color for each agent's path if desired
            path_line, = self.ax.plot([], [], 'o--', color=agent_color, alpha=0.6,
                                    markersize=5, linewidth=1.5, zorder=3,
                                    label=f'Agent {i} Path')
            self.path_lines.append(path_line)

        self.target_scatter = self.ax.scatter([], [], marker='X', color='gray',
                                            s=120, zorder=4)

        self.burn_box_artists = []
        if self.burn_box_mask is not None and self.burn_box_mask.any():
            expected_shape = (self.grid_size_y, self.grid_size_x)
            if self.burn_box_mask.shape != expected_shape:
                print(f"Warning: burn_box_mask shape {self.burn_box_mask.shape} does not match expected (Y,X) grid_size {expected_shape}. Resizing mask.")
                self.burn_box_mask = resize(
                    self.burn_box_mask,
                    output_shape=expected_shape,
                    order=0,
                    mode='constant',
                    cval=0.0,
                    anti_aliasing=False,
                    preserve_range=True
                ).astype(np.bool_)

            mask_overlay = self.ax.imshow(self.burn_box_mask, origin='lower',
                                        cmap='Blues', alpha=0.2, vmin=0, vmax=1, zorder=1,
                                        extent=imshow_extent)
            self.burn_box_artists.append(mask_overlay)
            
            dummy_line_burn_box, = self.ax.plot([], [], 's', color='blue', alpha=0.2, label='Burn Box Area', markersize=10)
            self.burn_box_artists.append(dummy_line_burn_box)

        else:
            print("No burn box mask supplied or mask is empty.")

        # Adjusted suptitle position to prevent overlap
        self.fig.suptitle("Firefighter Simulation Replay", fontsize=16, y=0.95)

        self.time_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                    color='white', fontsize=10,
                                    verticalalignment='bottom',
                                    bbox=dict(facecolor='black', alpha=0.6,
                                                edgecolor='none', pad=3))
        self.step_text = self.ax.text(0.02, 0.07, '', transform=self.ax.transAxes,
                                    color='white', fontsize=10,
                                    verticalalignment='bottom',
                                    bbox=dict(facecolor='black', alpha=0.6,
                                                edgecolor='none', pad=3))
        
        # Action Log Subplot setup
        self.ax_log.set_title("Action Log", fontsize=12, loc='left')
        self.ax_log.set_xticks([])
        self.ax_log.set_yticks([])
        self.ax_log.set_xlim(0, 1)
        self.ax_log.set_ylim(0, 1)
        self.ax_log.set_facecolor('#f0f0f0')
        self.ax_log.spines['top'].set_visible(False)
        self.ax_log.spines['right'].set_visible(False)
        self.ax_log.spines['bottom'].set_visible(False)
        self.ax_log.spines['left'].set_visible(False)

        self.action_log_text = self.ax_log.text(0.02, 0.98, '', transform=self.ax_log.transAxes,
                                            color='black', fontsize=9,
                                            verticalalignment='top', horizontalalignment='left',
                                            fontfamily='monospace', wrap=True)

        # Legend generation
        all_handles = []
        all_labels = []

        for patch in self.agent_patches:
            all_handles.append(patch)
            all_labels.append(patch.get_label())

        # CHANGED: Add all individual path lines to the legend
        # You might want to filter this if you only want a single 'Agent Path' label
        # For clarity, I've kept individual labels for each agent's path.
        for path_line in self.path_lines:
             all_handles.append(path_line)
             all_labels.append(path_line.get_label())

        self.target_legend_artists = []
        sorted_action_names = sorted(self.target_type_colors.keys())
        for action_name in sorted_action_names:
            if action_name == 'UnknownAction':
                continue
            color = self.target_type_colors.get(action_name)
            if color:
                dummy_artist, = self.ax.plot([], [], marker='X', linestyle='None',
                                            color=color, markersize=8, label=f'Target ({action_name})')
                self.target_legend_artists.append(dummy_artist)
                all_handles.append(dummy_artist)
                all_labels.append(dummy_artist.get_label())

        for artist in self.burn_box_artists:
            if artist.get_label() and 'Burn Box Area' in artist.get_label():
                all_handles.append(artist)
                all_labels.append(artist.get_label())
                break

        self.ax.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(1.05, 1.0),
                    fontsize=10, framealpha=0.9, borderpad=0.5, labelspacing=0.5)

        # Center both subplots on the x-axis by setting equal left and right margins
        self.fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.05, hspace=0.2)

        # Apply tight layout with padding to prevent title overlap
        self.fig.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)

        # Re-apply subplots_adjust after tight_layout to ensure centering
        self.fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.05, hspace=0.2)

        # Connect a resize event handler to adjust layout dynamically
        def on_resize(event):
            self.fig.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
            self.fig.subplots_adjust(left=0.25, right=0.75, top=0.92, bottom=0.05, hspace=0.2)
            self.fig.canvas.draw_idle()

        self._resize_id = self.fig.canvas.mpl_connect('resize_event', on_resize)

    def plot_action_locations(self, action_name: str, output_filepath: Optional[str] = None):
        """
        Creates a 2D grid plot showing the locations where a specific action type occurred.
        
        Args:
            action_name (str): The name of the action type (e.g., 'water', 'firebreak').
            output_filepath (Optional[str]): If provided, saves the plot to this file path.
                                            If None, displays the plot interactively.
        """
        if action_name not in self.action_type_map:
            print(f"Error: Action '{action_name}' not found in action_type_map.")
            return

        # Initialize a grid to count action occurrences
        action_grid = np.zeros((self.grid_size_y, self.grid_size_x), dtype=np.int32)
        action_id = self.action_type_map[action_name]

        # Count occurrences of the action at each location
        for frame_data in self.visualization_history:
            for target in frame_data['agent_targets']:
                if target is not None and target['type'] == action_id:
                    x, y = target['x'], target['y']
                    if 0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y:
                        action_grid[y, x] += 1

        # Create a new figure for the action location plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (grid cells)', fontsize=12)
        ax.set_ylabel('Y (grid cells)', fontsize=12)
        ax.set_xlim(-0.5, self.grid_size_x + 0.5)
        ax.set_ylim(-0.5, self.grid_size_y + 0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(np.arange(0, self.grid_size_x + 1, 5))
        ax.set_yticks(np.arange(0, self.grid_size_y + 1, 5))

        # Plot the action grid as a heatmap
        im = ax.imshow(action_grid, origin='lower', cmap='viridis',
                       vmin=0, vmax=max(1, np.max(action_grid)),
                       extent=[-0.5, self.grid_size_x + 0.5, -0.5, self.grid_size_y + 0.5])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label=f'{action_name} Action Count',
                            orientation='vertical', fraction=0.046, pad=0.10)
        cbar.ax.tick_params(labelsize=10)

        # Add burn box mask if available
        if self.burn_box_mask is not None and self.burn_box_mask.any():
            ax.imshow(self.burn_box_mask, origin='lower', cmap='Blues', alpha=0.2,
                      vmin=0, vmax=1,
                      extent=[-0.5, self.grid_size_x + 0.5, -0.5, self.grid_size_y + 0.5])
            ax.plot([], [], 's', color='blue', alpha=0.2, label='Burn Box Area', markersize=10)

        # Set title
        fig.suptitle(f"Locations of '{action_name}' Actions", fontsize=16, y=0.95)

        # Add legend
        if self.burn_box_mask is not None and self.burn_box_mask.any():
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=10)

        # Adjust layout
        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.05)

        if output_filepath:
            try:
                print(f"Saving action location plot to {output_filepath}...")
                plt.savefig(output_filepath, dpi=100, bbox_inches='tight')
                print(f"Plot saved to {output_filepath}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            finally:
                plt.close(fig)
        else:
            print(f"Displaying '{action_name}' action locations interactively.")
            plt.show()
            plt.close(fig)

    def update_plot(self, frame_data: Dict):
        """
        Updates the matplotlib plot with the given frame data.
        """
        self.im_fire.set_data(frame_data['fire_grid'])

        for i, pos in enumerate(frame_data['agent_positions']):
            if i < len(self.agent_patches) and pos is not None:
                self.agent_patches[i].set_center((pos[0] + 0.5, pos[1] + 0.5))
            elif i < len(self.agent_patches):
                # Move off-screen if agent position is None (e.g., terminated)
                self.agent_patches[i].set_center((-100, -100))

        target_x_coords = []
        target_y_coords = []
        target_colors = []

        for target_info in frame_data['agent_targets']:
            if target_info is not None and 'x' in target_info and 'y' in target_info:
                target_x_coords.append(target_info['x'] + 0.5)
                target_y_coords.append(target_info['y'] + 0.5)
                
                action_id = target_info['type']
                action_name = self._action_id_to_name_map.get(action_id, 'UnknownAction')
                target_colors.append(self.target_type_colors.get(action_name, self.target_type_colors['UnknownAction']))
        
        self.target_scatter.set_offsets(np.c_[target_x_coords, target_y_coords])
        self.target_scatter.set_color(target_colors)

        # CHANGED: Iterate and update each agent's path line
        for i, path_data in enumerate(frame_data['agent_paths']):
            if i < len(self.path_lines): # Ensure we have a corresponding path line object
                if path_data:
                    path_x = [p[0] + 0.5 for p in path_data]
                    path_y = [p[1] + 0.5 for p in path_data]
                    self.path_lines[i].set_data(path_x, path_y)
                else:
                    # Clear path if no path data (e.g., agent is idle or trapped)
                    self.path_lines[i].set_data([], [])


        self.time_text.set_text(f"Time: {frame_data.get('time', 'N/A'):.1f}s")
        self.step_text.set_text(f"Step: {frame_data['step']}")

        log_display_text = "\n" + "\n".join(frame_data.get('action_log', []))
        self.action_log_text.set_text(log_display_text)

        # CHANGED: Return all path lines for blit
        artists_to_return = [self.im_fire] + self.agent_patches + \
                            [self.target_scatter, self.time_text, self.step_text, self.action_log_text] + \
                            list(self.burn_box_artists) + \
                            self.path_lines # Add all path lines here!
        return artists_to_return


    def capture_frame_data(self, current_step: int, current_time: float,
                           agent_positions_grid: List[Optional[Tuple[int, int]]],
                           agent_target_intents: List[Optional[Dict]],
                           agent_current_paths: List[Optional[Deque]],
                           sim_active_fires_data: List[Dict],
                           sim_nx: int, sim_ny: int,
                           sim_dx: float, sim_dy: float
                          ) -> Dict:
        
        current_agent_positions = [pos for pos in agent_positions_grid]

        current_agent_targets = []
        for i in range(len(agent_target_intents)):
            intent = agent_target_intents[i]
            if intent is not None and 'x' in intent and 'y' in intent:
                current_agent_targets.append({'agent_idx': i, 'x': intent['x'], 'y': intent['y'], 'type': intent['type']})
                
                prev_intent = self._previous_agent_targets[i]
                if prev_intent is None or \
                   prev_intent.get('x') != intent['x'] or \
                   prev_intent.get('y') != intent['y'] or \
                   prev_intent.get('type') != intent['type']:
                    
                    action_name = self._action_id_to_name_map.get(intent['type'], 'UnknownAction')
                    log_entry = f"Step {current_step}: A{i} -> {action_name} at ({intent['x']},{intent['y']})"
                    self.action_log.append(log_entry)

            else:
                current_agent_targets.append(None) 
        
        self._previous_agent_targets = [target.copy() if target is not None else None for target in current_agent_targets]


        current_agent_paths_list = []
        for i in range(len(agent_current_paths)):
            if agent_current_paths[i] is not None:
                current_agent_paths_list.append(list(agent_current_paths[i]))
            else:
                current_agent_paths_list.append(None)

        fire_grid_for_viz = np.zeros((self.grid_size_y, self.grid_size_x), dtype=np.float32)
        
        block_x_ratio = self.grid_size_x / sim_nx
        block_y_ratio = self.grid_size_y / sim_ny

        if sim_active_fires_data is not None and len(sim_active_fires_data) > 0:
            for fire in sim_active_fires_data:
                phys_x_meters, phys_y_meters, _ = fire['position']
                
                phys_grid_x = int(np.round(phys_x_meters / sim_dx))
                phys_grid_y = int(np.round(phys_y_meters / sim_dy))

                agent_grid_x = int(phys_grid_x * block_x_ratio)
                agent_grid_y = int(phys_grid_y * block_y_ratio)

                if 0 <= agent_grid_x < self.grid_size_x and 0 <= agent_grid_y < self.grid_size_y:
                    fire_grid_for_viz[agent_grid_y, agent_grid_x] += fire['intensity']

        # Update global maximum intensity
        if fire_grid_for_viz.size > 0:
            self.global_max_intensity = max(self.global_max_intensity, np.max(fire_grid_for_viz))

        # Normalize fire_grid_for_viz so that the maximum intensity is 2
        if self.global_max_intensity > 0:
            fire_grid_for_viz = fire_grid_for_viz / self.global_max_intensity * 2
                
        return {
            'step': current_step,
            'time': current_time,
            'agent_positions': current_agent_positions,
            'agent_targets': current_agent_targets,
            'agent_paths': current_agent_paths_list,
            'fire_grid': fire_grid_for_viz,
            'action_log': list(self.action_log) 
        }

    def reset_history(self):
        """Clears the visualization history and closes any active plots."""
        self.visualization_history = []
        self.close()

    def add_frame(self, frame_data: Dict):
        """Adds a captured frame's data to the history."""
        self.visualization_history.append(frame_data)

    def get_last_frame_data(self) -> Optional[Dict]:
        """Returns the data for the last captured frame."""
        if self.visualization_history:
            return self.visualization_history[-1]
        return None

    def render_current_frame(self, mode: str = 'human'):
        """
        Renders the current state in a live plot.
        Called by env.render().
        """
        if not self.visualization_history:
            print("No visualization history to render.")
            return None if mode == 'rgb_array' else None

        if self.fig is None or self.ax is None:
            self._init_plot_artists()

        frame_data = self.get_last_frame_data()
        # CHANGED: update_plot now returns the list of artists to update for blitting.
        # We need to explicitly collect them here if we were using blitting in render_current_frame.
        # For 'human' mode, it's not strictly necessary to collect, but good practice.
        self.update_plot(frame_data)

        if mode == 'human':
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def animate_replay(self, output_filepath: Optional[str] = None, fps: int = 5):
        """
        Creates an animation from the captured visualization history.
        If output_filepath is None, the animation will pop up interactively.
        """
        if not self.visualization_history:
            print("No visualization data available to animate.")
            return

        print(f"Preparing animation with {len(self.visualization_history)} frames...")

        self._init_plot_artists()

        # Update with the first frame data to set initial positions/paths
        self.update_plot(self.visualization_history[0])

        def update(frame_idx):
            frame_data = self.visualization_history[frame_idx]
            # CHANGED: update_plot now returns the list of artists to update for blitting.
            artists_to_return = self.update_plot(frame_data)
            return artists_to_return

        anim = FuncAnimation(self.fig, update, frames=len(self.visualization_history),
                             interval=1000/fps, blit=True, repeat=True,
                             init_func=lambda: self.update_plot(self.visualization_history[0])) # init_func should also return artists

        if output_filepath:
            try:
                print(f"Saving replay to {output_filepath}...")
                anim.save(output_filepath, writer='ffmpeg', fps=fps, dpi=100)
                print(f"Replay saved to {output_filepath}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Please ensure FFmpeg is installed and in your system's PATH, or provide a .gif extension for Pillow writer.")
            finally:
                self.close()
        else:
            print("Displaying animation interactively. Close the window to continue.")
            plt.show()
            self.close()

    def close(self):
        """Closes any open matplotlib figures managed by this visualizer."""
        if self.fig:
            if self._resize_id is not None:
                self.fig.canvas.mpl_disconnect(self._resize_id)
                self._resize_id = None 

            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.ax_log = None 
            self.im_fire = None
            self.agent_patches = []
            self.target_scatter = None
            # CHANGED: Reset path_lines list
            self.path_lines = [] 
            self.burn_box_artists = []
            self.time_text = None
            self.step_text = None
            self.action_log_text = None 
            self.target_legend_artists = []
            self.action_log.clear() # Clear log
            self._previous_agent_targets = [None] * self.num_agents # Reset previous targets

if __name__ == "__main__":
    GRID_SIZE = (50, 50)
    NUM_AGENTS = 3
    SIM_DX = 2.0
    SIM_DY = 2.0
    
    ACTION_TYPE_MAP = {
        'water': 0, 'firebreak': 1, 'retardant': 2, 'air_support': 3, 'idle': 4
    } 

    SIM_NX = 100
    SIM_NY = 100
    dummy_sim_mask = np.zeros((SIM_NY, SIM_NX), dtype=np.bool_)
    dummy_sim_mask[20:80, 20:80] = True
    
    # Placeholder for your environment's mask creation.
    # Replace this with your actual environment object if needed.
    class DummyEnv:
        def create_agent_obs_mask(self, target_shape):
            # Example: create a simple mask for visualization purposes
            mask = np.zeros(target_shape, dtype=np.bool_)
            mask[10:40, 10:40] = True # A smaller burn box within the grid
            return mask

    dummy_env = DummyEnv() # Instantiate the dummy env
    burn_box_mask = dummy_env.create_agent_obs_mask(target_shape=GRID_SIZE)

    viz = FirefighterVisualizer(
        grid_size=GRID_SIZE,
        num_agents=NUM_AGENTS,
        burn_box_mask=burn_box_mask,
        sim_dx=SIM_DX,
        sim_dy=SIM_DY,
        action_type_map=ACTION_TYPE_MAP
    )

    total_steps = 200
    action_ids = list(ACTION_TYPE_MAP.values())
    for step in range(total_steps):
        current_time = step * 0.1

        fire_data_list = []
        if step % 10 == 0:
            new_fire_x_sim_cell = np.random.randint(0, GRID_SIZE[0])
            new_fire_y_sim_cell = np.random.randint(0, GRID_SIZE[1])
            new_fire_x_meters = new_fire_x_sim_cell * SIM_DX
            new_fire_y_meters = new_fire_y_sim_cell * SIM_DY
            fire_data_list.append({'position': (new_fire_x_meters, new_fire_y_meters, 0), 'intensity': 1.5})

        if step > 0 and viz.visualization_history and len(viz.visualization_history[-1]['fire_grid'].nonzero()[0]) > 0:
            prev_fire_grid_for_viz = viz.visualization_history[-1]['fire_grid']
            fire_y_coords, fire_x_coords = prev_fire_grid_for_viz.nonzero()
            
            for vy, vx in zip(fire_y_coords, fire_x_coords):
                intensity = prev_fire_grid_for_viz[vy, vx]
                if intensity > 0.1:
                    phys_x_meter = vx * SIM_DX
                    phys_y_meter = vy * SIM_DY

                    spread_x_meter = phys_x_meter + np.random.uniform(-SIM_DX, SIM_DX)
                    spread_y_meter = phys_y_meter + np.random.uniform(-SIM_DY, SIM_DY)
                    fire_data_list.append({'position': (spread_x_meter, spread_y_meter, 0), 'intensity': intensity * 0.8})

        agent_positions = []
        agent_targets = []
        agent_paths = []

        for i in range(NUM_AGENTS):
            px = int((GRID_SIZE[0] / 2) + 10 * np.sin(step * 0.1 + i * np.pi/3) + np.random.randint(-1, 2))
            py = int((GRID_SIZE[1] / 2) + 10 * np.cos(step * 0.1 + i * np.pi/3) + np.random.randint(-1, 2))
            px = np.clip(px, 0, GRID_SIZE[0] - 1)
            py = np.clip(py, 0, GRID_SIZE[1] - 1)
            agent_positions.append((px, py))

            # Simulate random target changes
            if step % 30 == 0 or np.random.rand() < 0.05: 
                target_x = np.random.randint(0, GRID_SIZE[0])
                target_y = np.random.randint(0, GRID_SIZE[1])
                action_id_for_target = action_ids[np.random.randint(0, len(action_ids))] 
                agent_targets.append({'type': action_id_for_target, 'x': target_x, 'y': target_y})
            else:
                agent_targets.append(None) 

            # Simulate different paths for each agent for testing
            current_path = deque()
            if np.random.rand() < 0.7: # Most agents have a path most of the time
                path_len = np.random.randint(2, 10)
                for k in range(path_len):
                    # Make paths slightly more distinct for visualization
                    path_x = agent_positions[i][0] + k * int(np.sin(i * np.pi/2 + step * 0.05))
                    path_y = agent_positions[i][1] + k * int(np.cos(i * np.pi/2 + step * 0.05))
                    current_path.append((np.clip(path_x, 0, GRID_SIZE[0]-1), np.clip(path_y, 0, GRID_SIZE[1]-1)))
            agent_paths.append(current_path if current_path else None)


        frame_data = viz.capture_frame_data(
            current_step=step,
            current_time=current_time,
            agent_positions_grid=agent_positions,
            agent_target_intents=agent_targets,
            agent_current_paths=agent_paths,
            sim_active_fires_data=fire_data_list,
            sim_nx=SIM_NX,
            sim_ny=SIM_NY,
            sim_dx=SIM_DX,
            sim_dy=SIM_DY
        )
        viz.add_frame(frame_data)

    viz.animate_replay(output_filepath="firefighter_simulation_replay_multiple_paths.mp4", fps=10)
    print("Visualization complete.")