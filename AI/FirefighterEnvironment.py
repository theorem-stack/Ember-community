import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
import heapq
from collections import deque
from scipy.ndimage import binary_dilation
from skimage.measure import block_reduce
from typing import Dict, Optional, Tuple, List, Any
from QuicFire.QuicFireEngine import QuicFireEngine
from QuicFire.QuicFireBurnBox import QuicFireBurnBox
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.RNGCollection import RNGCollection
from QuicFire.QuicFireEnvGenerator import ClimateConditions, generate_realistic_environment
from AI.DStarLite import DStarLite

class FirefighterEnvironment(gym.Env):
    """
    Custom environment for firefighter training, integrated with QuicFireSimulation.
    Uses 2D observation space with channels for fire, embers, temperature, fuel, moisture, wind, and ALL agent positions.
    Designed for a single network controlling multiple agents via action_idx, adapted for SMDP framework.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, grid_size: Tuple[int, int] = (20, 20), max_steps: int = 1000, num_agents: int = 1,
                 sim_base_config: Dict = None, module_seeds: RNGCollection = None, climate: ClimateConditions = None,
                 verbose: bool = True, enable_viz_recorder=True, enable_reward_logging: bool = True, is_training: bool = True):
        """
        Initializes the FirefighterEnvironment.
        """
        super().__init__()
        self.grid_size_x, self.grid_size_y = grid_size
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.sim_base_config = sim_base_config if sim_base_config is not None else {}
        self.verbose = verbose
        self.enable_viz_recorder = enable_viz_recorder
        self.enable_reward_logging = enable_reward_logging
        self.is_training = is_training
        self.random_seed = self.sim_base_config.get('random_seed', 42)
        self.module_seeds = module_seeds
        self.climate = climate

        # Validate sim_base_config
        required_sim_keys = ['nx', 'ny', 'nz', 'dx', 'dy', 'total_time']
        if not all(k in self.sim_base_config for k in required_sim_keys):
            raise ValueError(f"sim_base_config must contain all of: {required_sim_keys}")

        # Set target shape for downsampling
        self.target_shape = (self.grid_size_x, self.grid_size_y)

        # Define action space
        self.action_type_map: Dict[str, int] = {
            'water': 0, 'firebreak': 1, 'retardant': 2, 'air_support': 3, 'idle': 4
        }
        self.num_action_types = len(self.action_type_map)

        # Create the inverse map once during initialization
        self.action_id_to_name_map: Dict[int, str] = {v: k for k, v in self.action_type_map.items()}

        # Separate actions by type
        self.local_actions = {'water', 'firebreak', 'retardant'}
        self.global_actions = {'air_support', 'idle'}

        # Track global actions
        self.agent_global_action_timer: Dict[int, Optional[int]] = {idx: None for idx in range(num_agents)}

        # Action costs
        self.action_costs: Dict[str, float] = {
            'water': QuicFireConstants.WATER_COST,
            'firebreak': QuicFireConstants.FIREBREAK_COST,
            'retardant': QuicFireConstants.RETARDANT_COST,
            'air_support': QuicFireConstants.AIR_SUPPORT_COST,
        }

        # Define action ranges
        self.action_ranges: Dict[str, float] = {
            'water': QuicFireConstants.WATER_ACTION_RANGE,
            'firebreak': QuicFireConstants.FIREBREAK_ACTION_RANGE,
            'retardant': QuicFireConstants.RETARDANT_ACTION_RANGE,
            'air_support': QuicFireConstants.AIR_SUPPORT_ACTION_RANGE,
        }

        self.action_z_ranges: Dict[str, int] = {
            'water': QuicFireConstants.WATER_ACTION_Z_RANGE,
            'firebreak': QuicFireConstants.FIREBREAK_ACTION_Z_RANGE,
            'retardant': QuicFireConstants.RETARDANT_ACTION_Z_RANGE,
            'air_support': QuicFireConstants.AIR_SUPPORT_ACTION_Z_RANGE,
        }

        # Resource capacities
        self.resource_capacity: Dict[str, float] = {
            'water': QuicFireConstants.WATER_CAPACITY,
            'retardant': QuicFireConstants.RETARDANT_CAPACITY,
            'air_support': QuicFireConstants.AIR_SUPPORT_CAPACITY,
        }

        self.global_action_delays: Dict[str, float] = {
            'idle': QuicFireConstants.IDLE_DELAY_STEPS,
            'air_support': QuicFireConstants.AIR_SUPPORT_DELAY_STEPS
        }

        # Define observation space
        self.observation_channels: Dict[int, str] = {
            0: 'heat_flux', 1: 'energy_packets', 2: 'temperature', 3: 'fuel_density',
            4: 'moisture_fraction', 5: 'wind_magnitude', 6: 'agent_positions', 7: 'burn_box',
            8: 'placeholder'
        }
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size_x, self.grid_size_y, len(self.observation_channels)), dtype=np.float32
        )

        # Constant
        self.max_heat_flux_per_cell = QuicFireConstants.HEAT_FLUX * self.sim_base_config['dx'] * self.sim_base_config['dy']

        # Initialize environment and agents
        self._setup_environment()
        self._initialize_agents()
        self._initialize_pathfinding()
        self._initialize_calculated_vars()

        # Debugging visual
        if self.enable_viz_recorder:
            # Conditional import
            from AI.Debugging.FirefighterVisualizer import FirefighterVisualizer
            from AI.Debugging.HelperFuns import plot_mask
            
            self.viz_recorder = FirefighterVisualizer(
                grid_size=grid_size,
                num_agents=num_agents,
                burn_box_mask=self.burn_box_obs_mask,
                sim_dx=sim_base_config['dx'], 
                sim_dy=sim_base_config['dy'], 
                action_type_map=self.action_type_map
            )

        # Debugging rewards
        if self.enable_reward_logging:
            self.episode_reward_log: List[Dict[str, Any]] = []
            self.per_agent_reward_components_current_step: List[Dict[str, float]] = []
            self.global_reward_components_current_step: Dict[str, float] = {}

    def _initialize_reward_logs_for_step(self) -> None:
        """Initializes reward component dictionaries for the current step."""
        self.per_agent_reward_components_current_step = [
            {'total_reward': 0.0, 'fire_reduction': 0.0,
             'movement_cost': 0.0, 'invalid_action_penalty': 0.0 }
            for _ in range(self.num_agents)
        ]
        self.global_reward_components_current_step = {
            'total_reward': 0.0,
            'active_fire_outside_bb_penalty': 0.0,
            'fuel_consumed_outside_bb_penalty': 0.0,
            'fuel_consumed_inside_bb_reward': 0.0,
            'water_usage_penalty': 0.0,
            'retardant_usage_penalty': 0.0,
            'air_support_usage_penalty': 0.0,
            'time_penalty': 0.0,
            'mission_success_reward': 0.0,
            'mission_failure_spread_penalty': 0.0,
            'mission_failure_timeout_penalty': 0.0,
        }

    def _log_step_rewards(self, global_reward: float) -> None:
        """Appends the current step's reward data to the episode log."""
        step_log_entry = {
            'step': self.current_step,
            'global_reward': global_reward,
            'global_reward_components': self.global_reward_components_current_step,
            'agent_rewards': self.per_agent_reward_components_current_step
        }
        self.episode_reward_log.append(step_log_entry)

    def _initialize_calculated_vars(self):
        self._pre_action_fuel_density = None
        self._pre_action_heat_flux = None
        self._pre_action_agent_obs_heat_flux = None
        self._current_action_agent_obs_heat_flux = None
        self._current_action_agent_obs_moisture_fraction = None
        
    def _setup_environment(self):
        """Sets up the environment, burn box, and simulation from scratch."""
        if self.verbose:
            print("Setting up environment and simulation...")

        # Generate a new environment
        self.environment = generate_realistic_environment(
            nx=self.sim_base_config['nx'],
            ny=self.sim_base_config['ny'],
            dx=self.sim_base_config['dx'],
            dy=self.sim_base_config['dy'],
            max_elevation=QuicFireConstants.MAX_TERRAIN_ELEVATION,
            climate=self.climate,
            terrain_rng=self.module_seeds.get_rng('terrain'),
            vegetation_rng=self.module_seeds.get_rng('vegetation'),
        )
        self.height_map = self.environment['height_map']
        self.veg_map = self.environment['vegetation_map']
        self.fuel_map = self.environment['fuel_map']
        self.fuel_moisture_map = self.environment['fuel_moisture_map']

        # Create a new burn box
        self.burn_box = QuicFireBurnBox(
            rng=self.module_seeds.get_rng('burn_box'),
            nx=self.sim_base_config['nx'], 
            ny=self.sim_base_config['ny'],
            dx=self.sim_base_config['dx'],
            dy=self.sim_base_config['dy'],
            ignition_inner_buffer=QuicFireConstants.IGNITION_INNER_BUFFER
        )
        self.burn_box_mask_sim = self.burn_box.get_sim_mask()
        self.burn_box_obs_mask = self.burn_box.create_agent_obs_mask(self.target_shape)

        # Initialize a new simulation
        self.sim = QuicFireEngine(
            config=self.sim_base_config,
            veg_map=self.veg_map, 
            fuel_map=self.fuel_map, 
            fuel_moisture_map=self.fuel_moisture_map,
            height_map=self.height_map,
            module_seeds=self.module_seeds,
            burn_box=self.burn_box
        )

        # Set initial diagnostics and fuel data
        self.initial_diagnostics = self.sim.get_diagnostics()
        self.initial_fuel_per_column_2d = np.sum(self.sim.fuel_density, axis=-1)
        self.initial_fuel_outside_bb = sum(self.initial_fuel_per_column_2d[~self.burn_box_mask_sim])
        self.initial_fuel_inside_bb = sum(self.initial_fuel_per_column_2d[self.burn_box_mask_sim])

    def _initialize_agents(self):
        """Initializes or resets agent-specific attributes."""
        self.current_step = 0
        self.agent_pos: List[Tuple[int, int]] = self._initialize_agent_positions(self.num_agents, random_seed=self.random_seed)
        self.current_water: List[float] = [self.resource_capacity['water'] for _ in range(self.num_agents)]
        self.current_retardant: List[float] = [self.resource_capacity['retardant'] for _ in range(self.num_agents)]
        self.current_air_support: List[float] = [self.resource_capacity['air_support'] for _ in range(self.num_agents)]
        self.agent_target_intent: List[Optional[Dict]] = [None for _ in range(self.num_agents)]
        self.agent_current_path: List[Optional[deque]] = [None for _ in range(self.num_agents)]
        self.agent_action_invalid_due_to_resources: List[bool] = [False for _ in range(self.num_agents)]
        self.last_movement_cost_per_agent: List[float] = [0.0 for _ in range(self.num_agents)]
        self.option_rewards: List[float] = [0.0] * self.num_agents
        self.option_terminated: List[bool] = [False] * self.num_agents
        self.agent_applied_action: List[bool] = [False] * self.num_agents
        if self.verbose:
            print(f"Initialized {self.num_agents} agents with positions: {self.agent_pos}")

    def _initialize_pathfinding(self):
        # --- Initialize one D* Lite instance per agent ---
        self.agent_d_star_planners = []
        for _ in range(self.num_agents):
            self.agent_d_star_planners.append(
                DStarLite(
                    grid_shape=(self.grid_size_x, self.grid_size_y),
                    get_cost_func=self.get_cell_cost,          
                    get_neighbors_func=self.get_grid_neighbors
                )
            )

    def get_cell_cost(self, from_node: tuple, to_node: tuple) -> float:
        """
        D* Lite Helper Function: Calculates the cost of moving from from_node to to_node based on heat_flux.
        """
        f_r, f_c = from_node
        t_r, t_c = to_node
        
        # Ensure coordinates are within bounds
        if not (0 <= t_r < self.grid_size_x and 0 <= t_c < self.grid_size_y):
            return float('inf') # Impassable (out of bounds)

        # Get raw heat_flux data from the current simulation state
        heat_flux = self._current_action_agent_obs_heat_flux[t_r, t_c] 

        # If heat_flux exceeds a threshold, it's impassable.
        if heat_flux > QuicFireConstants.HEAT_FLUX_DANGER_THRESHOLD:
            return float('inf') 

        # Calculate base cost (including diagonal if applicable)
        base_cost = 1

        # Diagonal move
        if abs(f_r - t_r) + abs(f_c - t_c) == 2:
            base_cost *= math.sqrt(2) 

        # Burn box movement
        if self.burn_box_obs_mask[f_r, f_c]:
            base_cost += QuicFireConstants.BURN_BOX_MOVEMENT_COST 

        return base_cost

    def get_grid_neighbors(self, node: tuple) -> list:
        """
        D* Lite Helper Function: Returns a list of valid (in-bounds) 8-directional neighbors for a given node.
        Adjust for 4-directional if your agents only move orthogonally.
        """
        r, c = node
        neighbors = []
        # 8-directional movement
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: # Skip the current node itself
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size_x and 0 <= nc < self.grid_size_y:
                    # Optional: Add checks for static obstacles here (e.g., mountains, water bodies)
                    # if self.sim.terrain_type[nr, nc] not in [TERRAIN_OBSTACLE_TYPES]:
                    neighbors.append((nr, nc))
        return neighbors
    
    def _get_safe_pathfinding_target(self, action_target_node: Tuple[int, int], action_range: int) -> Optional[Tuple[int, int]]:
        """
        Finds the closest non-fire (safe) cell to the action_target_node
        within the specified action_range that is suitable for pathfinding.
        """
        target_r, target_c = action_target_node
        grid_x, grid_y = self.grid_size_x, self.grid_size_y
        danger_threshold = QuicFireConstants.HEAT_FLUX_DANGER_THRESHOLD

        # 1. Define the search window (bounding box) around the target
        r_min = max(0, target_r - action_range)
        r_max = min(grid_x - 1, target_r + action_range)
        c_min = max(0, target_c - action_range)
        c_max = min(grid_y - 1, target_c + action_range)

        # Slice the heat_flux matrix for the relevant region
        search_area_heat_flux = self._current_action_agent_obs_heat_flux[r_min : r_max + 1, c_min : c_max + 1]

        # 2. Binary mask the cells: Identify safe cells within the search area
        safe_mask = search_area_heat_flux <= danger_threshold

        # 3. Check if all cells in the search area are unsafe
        if not np.any(safe_mask):
            return None

        # 4. Get the coordinates of all safe cells within the search area
        relative_safe_rows, relative_safe_cols = np.where(safe_mask)

        # Convert relative coordinates back to absolute grid coordinates
        absolute_safe_cells = []
        for r_rel, c_rel in zip(relative_safe_rows, relative_safe_cols):
            absolute_safe_cells.append((r_min + r_rel, c_min + c_rel))

        # 5. Calculate distances and return the closest
        min_distance_sq = float('inf')
        closest_safe_cell = None

        for safe_r, safe_c in absolute_safe_cells:
            distance = max(abs(safe_r - target_r), abs(safe_c - target_c))

            if distance < min_distance_sq:
                min_distance_sq = distance
                closest_safe_cell = (safe_r, safe_c)
            elif distance == min_distance_sq:
                # Tie-breaking rule if multiple cells have the same minimum distance.
                pass

        return closest_safe_cell

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the simulation and environment state for a new episode.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.random_seed = seed
        self.sim.close()
        self._setup_environment()
        self._initialize_agents()
        self._initialize_pathfinding()
        self._initialize_calculated_vars()
        self.episode_reward_log = [] # Clear the full log for a new episode
        self.per_agent_reward_components_current_step = [] 
        self.global_reward_components_current_step = {}
        print("Environment reset completed.")
        return self._get_observation()

    def step(self, actions_for_all_agents: np.ndarray) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """
        Advance simulation, process agent intents and movements, and compute per-agent rewards using SMDP framework.
        
        Note: This function now has a streamlined return signature for better
        compatibility with standard Gym-like environments.
        """

        # --- Phase 1: Environment Setup & Pre-action State Capture ---
        # Only perform reward logging if it's explicitly enabled (good practice)
        if self.enable_reward_logging:
            self._initialize_reward_logs_for_step()

        self.option_terminated = [False] * self.num_agents
        # Using list comprehensions for efficient initialization
        self.last_movement_cost_per_agent = [0.0 for _ in range(self.num_agents)]
        self.agent_action_invalid_due_to_resources = [False for _ in range(self.num_agents)]
        
        # Store simulation states (deep copies for accurate reward calculation)
        # Note: If these copies are of very large grids, they can be a performance bottleneck.
        # Consider if you can calculate rewards without deep copies if performance becomes an issue.
        self._pre_action_fuel_density = self.sim.fuel_density.copy()
        self._pre_action_heat_flux = self.sim.heat_flux.copy()
        self._pre_action_agent_obs_heat_flux = self._downsample_channel(self._pre_action_heat_flux, self.target_shape, np.max)

        # Capture intents before they are modified
        intents_before = [intent.copy() if intent is not None else None for intent in self.agent_target_intent]

        # --- Phase 2: Process Agent Intents and Execute Actions ---
        for action_tuple in actions_for_all_agents:
            agent_idx, chosen_action_type, chosen_x_grid, chosen_y_grid = action_tuple.astype(int)

            if self.agent_target_intent[agent_idx] is None:
                self.agent_target_intent[agent_idx] = {
                    'type': chosen_action_type,
                    'x': chosen_x_grid,
                    'y': chosen_y_grid
                }
                self.agent_current_path[agent_idx] = None
                
                if self.verbose:
                    print(f"Agent {agent_idx}: New intent set to ({chosen_x_grid}, {chosen_y_grid}). Path will be planned on next update.")

        # --- Phase 3: Advance Simulation and Check Early Termination Conditions ---
        self.sim.step()
        self.current_step += 1

        self._current_action_agent_obs_heat_flux = self._downsample_channel(self.sim.heat_flux, self.target_shape, np.max)
        self._current_action_agent_obs_moisture_fraction = self._downsample_channel(self.sim.moisture_fraction, self.target_shape, np.mean)

        # --- D* Lite: Notify planners of Environment Changes ---
        # This nested loop can be a performance bottleneck with many agents and frequent changes.
        # Consider a more efficient batch update method for D* Lite if this becomes an issue.
        changed_cells = self._detect_environment_changes()
        if changed_cells:
            if self.verbose:
                print(f"Detected {len(changed_cells)} environment changes. Notifying D* Lite planners.")
            for agent_d_star_planner in self.agent_d_star_planners:
                for r, c in changed_cells:
                    agent_d_star_planner.update_edge_cost((r, c), (r, c))

        for i in range(self.num_agents):
            self._update_agent_state_and_act(i)

        # --- Phase 4: Calculate Rewards and Update Info ---
        per_agent_rewards = self._calculate_per_agent_rewards(intents_before=intents_before)
        info = self.sim.get_diagnostics()
        for i in range(self.num_agents):
            if self.agent_target_intent[i] is not None or self.option_terminated[i]:
                self.option_rewards[i] += per_agent_rewards[i]
            
            if self.option_terminated[i]:
                info[f'agent_{i}_option_reward'] = self.option_rewards[i]
                self.option_rewards[i] = 0.0

        # --- Phase 5: Determine Episode Termination ---
        obs = self._get_observation()
        done = self._check_done()
        info['global_reward'] = self._compute_global_reward(done)

        # ----------------------------------------------------------------------
        # --- Phase 6: Optimization - Conditional Visualization Data Capture ---
        # This entire block is now wrapped in a check for training mode.
        # This is a critical optimization for performance during RL training.
        if not self.is_training:
            visualization_info: Dict[str, Any] = {}
            visualization_info.update(self.sim.get_xr_visualization_data())

            agent_data_list = []
            for i in range(self.num_agents):
                current_path_list = list(self.agent_current_path[i]) if self.agent_current_path[i] is not None else []
                current_intent = self.agent_target_intent[i] if self.agent_target_intent[i] is not None else {}
                current_intent = {k: v.item() if isinstance(v, np.generic) else v for k, v in current_intent.items()}
                
                ### Convert all position coordinates from Agent Grid to Sim Grid

                # Update Target is a range of Sim Grid cells
                if current_intent:
                    intent_x, intent_y = current_intent['x'], current_intent['y']
                    current_intent['sim_x_start'], current_intent['sim_x_end'], current_intent['sim_y_start'], current_intent['sim_y_end'] = self._get_sim_coords(intent_x, intent_y)

                # Unity Flag: indicate that action was performed
                current_intent['agent_applied_action'] = self.agent_applied_action[i]

                # Agent Position as specific Sim Grid cell (simplified assumption as corner cell)
                agent_x, agent_y = self.agent_pos[i]
                sim_x_start, _, sim_y_start, _ = self._get_sim_coords(agent_x, agent_y)
                agent_position = (sim_x_start, sim_y_start)

                # Convert path positions as specific Sim Grid cells (simplified assumption as corner cell)
                sim_grid_path = []
                for pos in current_path_list:
                    sim_x_start, _, sim_y_start, _ = self._get_sim_coords(pos[0], pos[1])
                    sim_grid_path.append((sim_x_start, sim_y_start))

                # Store visualization data
                agent_data = {
                    'agent_id': i,
                    '_sim':{
                        'position': agent_position,
                        'path': sim_grid_path,
                    },
                    '_ai':{
                        'position': self.agent_pos[i],
                        'path': current_path_list,
                    },
                    'intent': current_intent,
                    'resources': {
                        'water': self.current_water[i],
                        'retardant': self.current_retardant[i],
                        'air_support': self.current_air_support[i]
                    }
                }
                agent_data_list.append(agent_data)
            
            visualization_info['agents'] = agent_data_list
            visualization_info['step_count'] = self.current_step
            visualization_info['time_elapsed'] = self.current_step * QuicFireConstants.DT
            visualization_info['grid_metadata'] = {
                'nx': self.sim_base_config['nx'],
                'ny': self.sim_base_config['ny'],
                'dx': self.sim_base_config['dx'],
                'dy': self.sim_base_config['dy'],
                'z_levels': self.sim.z_levels.tolist(),
                'dz': self.sim.dz.tolist(),
                'agent_obs_x': self.grid_size_x, 
                'agent_obs_y': self.grid_size_y
            }
            visualization_info['agent_obs_heat_flux'] = self._current_action_agent_obs_heat_flux.tolist()

            info['viz_data'] = visualization_info

            # DEBUG: Temporary
            # Reset Unity flag indicating that agent has performed action
            self.agent_applied_action = [False] * self.num_agents

        if self.enable_reward_logging:
            self._log_step_rewards(info['global_reward'])

        if self.enable_viz_recorder:
            frame_data = self.viz_recorder.capture_frame_data(
                current_step=self.current_step, 
                current_time=self.current_step * QuicFireConstants.DT,
                agent_positions_grid=self.agent_pos,
                agent_target_intents=self.agent_target_intent,
                agent_current_paths=self.agent_current_path,
                sim_active_fires_data=self.sim.get_xr_visualization_data().get('fires', []),
                sim_nx=self.sim_base_config['nx'], 
                sim_ny=self.sim_base_config['ny'], 
                sim_dx=self.sim_base_config['dx'], 
                sim_dy=self.sim_base_config['dy'], 
            )
            self.viz_recorder.add_frame(frame_data)
        # ----------------------------------------------------------------------

        return obs, per_agent_rewards, done, info
    
    def _detect_environment_changes(self) -> list[tuple]:
        """
        Compares current sim state with previous to find cells whose costs have changed.
        This uses vectorized NumPy operations for efficiency.
        """
        # Calculate the absolute difference between current and previous heat flux
        heat_diff = np.abs(self._current_action_agent_obs_heat_flux - self._pre_action_agent_obs_heat_flux)

        # Condition 1: Significant change in heat flux
        cond1 = heat_diff > 1.0

        # Condition 2: Transition across the impassable threshold
        # Cells that were passable (<= threshold) and are now impassable (> threshold)
        cond2_transition_to_impassable = (self._pre_action_agent_obs_heat_flux <= QuicFireConstants.HEAT_FLUX_DANGER_THRESHOLD) & \
                                        (self._current_action_agent_obs_heat_flux > QuicFireConstants.HEAT_FLUX_DANGER_THRESHOLD)

        # Combine the conditions using logical OR
        # Any cell satisfying either condition is considered "changed"
        changed_mask = cond1 | cond2_transition_to_impassable

        # Get the coordinates (row, col) of the True values in the boolean mask
        # np.where returns two arrays (rows, cols) for a 2D array
        rows, cols = np.where(changed_mask)

        # Combine them into a list of tuples
        changed_cells = list(zip(rows, cols))

        return changed_cells
    
    def _update_agent_state_and_act(self, agent_idx: int):
        """
        Manages an individual agent's movement or action application within an SMDP option.
        Prioritizes danger avoidance, handles local and global actions differently,
        and terminates options upon completion.
        """

        # --- 1. Check for Terminal State ---
        # If the agent has no assigned option, there's nothing to do.
        if self.agent_target_intent[agent_idx] is None:
            if self.verbose:
                print(f"Agent {agent_idx}: No active option. Terminated.")
            self.option_terminated[agent_idx] = True
            return

        # --- Get Current Action Details ---
        current_agent_x, current_agent_y = self.agent_pos[agent_idx]
        target_intent = self.agent_target_intent[agent_idx]
        target_x = target_intent['x']
        target_y = target_intent['y']
        action_type = target_intent['type']
        action_name = self.action_id_to_name_map.get(action_type)

        # --- 2. Handle Global Actions (No Movement, Delay-Based) ---
        if action_name in self.global_actions:
            # Check if the timer exists and has started.
            timer_value = self.agent_global_action_timer.get(agent_idx)
            
            if timer_value is None:
                # First time this global action is encountered.
                if self.verbose:
                    print(f"Agent {agent_idx}: Initiating global action '{action_name}'.")
                self._initiate_global_action(agent_idx, action_name)
                # Global actions do not require movement, so clear the path.
                self.agent_current_path[agent_idx] = deque()
                # The timer is now set by _initiate_global_action.
                # Do not proceed further this step; the agent is now "waiting".
                return
            
            # Action has been initiated, now manage the timer.
            if timer_value > 0:
                self.agent_global_action_timer[agent_idx] -= 1
                if self.verbose:
                    print(f"Agent {agent_idx}: Waiting for '{action_name}' completion. Timer: {self.agent_global_action_timer[agent_idx]}")
                # The agent is still waiting, so check position for safety and adjust if necessary.
                if self._is_agent_in_danger(agent_idx):
                    self._handle_no_path_or_danger(agent_idx, current_agent_x, current_agent_y)
                return
            else: # timer_value is 0 or less
                # The timer has run out. The action is complete.
                if self.verbose:
                    print(f"Agent {agent_idx}: Global action '{action_name}' has completed. Applying effect and terminating option.")
                self._apply_action_at_target(agent_idx, action_name, target_x, target_y)
                
                # Use pop for safer removal of the timer key.
                self.agent_global_action_timer.pop(agent_idx, None)

                self.agent_target_intent[agent_idx] = None
                self.agent_current_path[agent_idx] = None
                self.option_terminated[agent_idx] = True
                return

        # --- 3. Handle Local Actions (Movement-Based) ---
        elif action_name in self.local_actions:
            # Check if agent has reached the target or is in range to apply the action.
            is_at_target_or_in_range = False
            if (current_agent_x, current_agent_y) == (target_x, target_y):
                is_at_target_or_in_range = True
            else:
                is_at_target_or_in_range = self._is_within_action_range(agent_idx, target_x, target_y, action_name)

            if is_at_target_or_in_range:
                # Agent is at or within range, apply the action and terminate the option.
                self._apply_action_at_target(agent_idx, action_name, target_x, target_y)
                self.agent_target_intent[agent_idx] = None
                self.agent_current_path[agent_idx] = None
                self.option_terminated[agent_idx] = True

                # Flag for Unity
                self.agent_applied_action[agent_idx] = True
                return

            # Agent is not at target, needs to move.
            agent_d_star_planner = self.agent_d_star_planners[agent_idx]
            agent_d_star_planner.set_start(self.agent_pos[agent_idx])
            
            goal_node = self._get_safe_pathfinding_target((target_x, target_y), self.action_ranges[action_name])
            
            if goal_node:
                agent_d_star_planner.set_goal(goal_node) 
                agent_d_star_planner.compute_shortest_path() 
                new_path_list = agent_d_star_planner.get_path(self.agent_pos[agent_idx], goal_node)
            else:
                new_path_list = None
            
            # --- Path Execution ---
            if new_path_list and len(new_path_list) > 1:
                # Update and move along the path
                self.agent_current_path[agent_idx] = deque(new_path_list)
                if self.agent_current_path[agent_idx][0] == self.agent_pos[agent_idx]:
                    self.agent_current_path[agent_idx].popleft()
                
                self.agent_pos[agent_idx] = self.agent_current_path[agent_idx].popleft()
                self.last_movement_cost_per_agent[agent_idx] = QuicFireConstants.MOVEMENT_PENALTY
            else:
                # No path found to the local target. This is a failure state.
                if self.verbose:
                    print(f"Agent {agent_idx}: No path found to target. Terminating option.")
                self.agent_target_intent[agent_idx] = None
                self.agent_current_path[agent_idx] = None
                self.option_terminated[agent_idx] = True

        # --- 4. Handle Unrecognized Actions ---
        else:
            if self.verbose:
                print(f"Agent {agent_idx}: Unrecognized action type '{action_name}'. Terminating option.")
            self.agent_target_intent[agent_idx] = None
            self.agent_current_path[agent_idx] = None
            self.option_terminated[agent_idx] = True

    def _is_agent_in_danger(self, agent_idx: int) -> bool:
        """Checks if the agent's current cell is dangerously hot."""
        x, y = self.agent_pos[agent_idx]
        return self._current_action_agent_obs_heat_flux[x, y] >= QuicFireConstants.HEAT_FLUX_DANGER_THRESHOLD

    def _initiate_global_action(self, agent_idx: int, action_name: str):
        """Calculates and sets the delay timer for a new global action."""
        # You can define a specific delay for each global action here
        delay = self.global_action_delays[action_name]
        if not delay:
            delay = 0
            
        self.agent_global_action_timer[agent_idx] = delay

    def _handle_no_path_or_danger(self, agent_idx: int, current_agent_x: int, current_agent_y: int):
        """
        Helper function to handle scenarios where no path is found or the next step is dangerous.
        Attempts to find an escape path, otherwise terminates the option.
        """
        if self.option_terminated[agent_idx]:
            return

        current_agent_pos = (current_agent_x, current_agent_y)

        # Find the escape path using the integrated search
        escape_path_list = self._find_nearest_safe_zone(current_agent_pos, agent_idx)
        
        if escape_path_list:
            # The returned path includes the current position as the start. Remove it.
            if escape_path_list[0] == current_agent_pos:
                escape_path_list.pop(0)

            if len(escape_path_list) > 0:
                # Adjust agent path without resetting the agent's intent or target position.
                # If environment changes and allows agent to re-engage the original target it will.
                self.agent_current_path[agent_idx] = deque(escape_path_list)
                
                if self.verbose:
                    print(f"Agent {agent_idx}: Escape path found with {len(self.agent_current_path[agent_idx])} steps.")
                
                # Move along the escape path
                next_pos = self.agent_current_path[agent_idx].popleft()
                self.agent_pos[agent_idx] = next_pos
                self.last_movement_cost_per_agent[agent_idx] = QuicFireConstants.MOVEMENT_PENALTY
                
            else:
                # Agent is already at the first safe zone found
                if self.verbose:
                    print(f"Agent {agent_idx}: Already in a safe zone. No immediate move needed.")
                self.agent_current_path[agent_idx] = deque() 
                self.agent_target_intent[agent_idx] = None # Agent has escaped, clear its previous mission
                # You might want to introduce a 'safe' state for the agent here.

        else: # No suitable safe zone found by _find_nearest_safe_zone
            if self.verbose:
                print(f"Agent {agent_idx}: TRAPPED! No safe zone found to escape to. Option terminated.")
            self.agent_current_path[agent_idx] = deque() # Empty path
            self.agent_target_intent[agent_idx] = None
            self.option_terminated[agent_idx] = True

    def _find_nearest_safe_zone(self, agent_current_pos_grid: Tuple[int, int], agent_idx: int) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the nearest safe zone from the agent's current position using a single A*/Dijkstra search.
        The goal is dynamic: the first reachable cell on the downsampled moisture grid
        that meets the moisture safety threshold.
        
        Args:
            agent_current_pos_grid (Tuple[int, int]): Agent's current (row, col) in the full-res grid.
            agent_idx (int): The index of the agent.

        Returns:
            Optional[List[Tuple[int, int]]]: The path (list of (row, col) tuples) to the nearest safe zone,
                                              or None if no safe zone is found within search limits.
        """
        # Initialize A* (or Dijkstra if heuristic is not used) elements.
        # Priority queue: (f_score, cost_g, node)
        # f_score = cost_g + heuristic (if using A*)
        # cost_g = actual cost from start to current_node
        
        # For this "any goal" search, a simple Dijkstra (A* with h=0) is usually best
        # unless you have a heuristic that guides towards "any sufficiently moist cell".
        # Let's stick with Dijkstra principles here for clarity and robust "nearest by path cost".
        
        pq = [(0, agent_current_pos_grid)] # (cost_g, node)
        cost_so_far = {agent_current_pos_grid: 0}
        came_from = {agent_current_pos_grid: None} # To reconstruct path

        # Limit search cost to avoid exploring the entire map indefinitely
        max_search_cost_for_escape = QuicFireConstants.MAX_ESCAPE_SEARCH_COST

        # Use percentile to prioritize safest cells in changing environment
        flat_moisture = self._current_action_agent_obs_moisture_fraction.flatten()
        _current_dynamic_moisture_threshold = np.percentile(flat_moisture, QuicFireConstants.MOISTURE_PERCENTILE_THRESHOLD)

        while pq:
            current_cost_g, current_node = heapq.heappop(pq)

            if current_cost_g > cost_so_far.get(current_node, float('inf')):
                continue # Already found a cheaper path

            if current_cost_g > max_search_cost_for_escape:
                continue # Reached search limit

            # --- Check if current_node qualifies as a "safe zone" based on moisture ---
            # No need for macro_r, macro_c calculation as _downsampled_moisture_map
            # now has the same dimensions as the pathfinding grid.
            r, c = current_node[0], current_node[1]
            
            # Ensure coordinates are within the bounds of the downsampled moisture map
            # This is important if your _downsample_channel pads or crops.
            if not (0 <= r < self.grid_size_x and \
                    0 <= c < self.grid_size_y):
                continue # Node outside bounds of moisture map (shouldn't happen often if dims match)

            moisture_content = self._current_action_agent_obs_moisture_fraction[r, c]

            # Direct check for sufficient moisture
            if moisture_content >= _current_dynamic_moisture_threshold:

                # Found the nearest (cost-wise) safe zone that meets moisture criteria!
                if self.verbose:
                    print(f"Agent {agent_idx}: Found escape target at {current_node} "
                          f"(path cost: {current_cost_g:.2f}, moisture: {moisture_content:.2f})")
                
                # Reconstruct and return the path
                path = []
                temp_node = current_node
                while temp_node is not None:
                    path.append(temp_node)
                    temp_node = came_from[temp_node]
                return path[::-1] # Path from start to target

            # Explore neighbors (using D* Lite's internal _get_cost and _get_neighbors)
            # D* Lite's cost function _get_path_cost already incorporates current heat flux.
            for neighbor in self.get_grid_neighbors(current_node):
                move_cost = self.get_cell_cost(current_node, neighbor) 

                if move_cost == float('inf'): # Impassable due to CURRENT fire (or other obstacles)
                    continue

                new_cost_g = current_cost_g + move_cost

                if new_cost_g < cost_so_far.get(neighbor, float('inf')):
                    cost_so_far[neighbor] = new_cost_g
                    came_from[neighbor] = current_node
                    heapq.heappush(pq, (new_cost_g, neighbor))

        if self.verbose:
            print(f"Agent {agent_idx}: No suitable safe zone found within search limits.")
        return None # No safe zone found within specified parameters

    def _check_done(self) -> bool:
        """
        Check termination conditions for the episode.
        """
        diagnostics = self.sim.get_diagnostics()
        fire_out = diagnostics['active_fires'] == 0
        
        # Check if fuel is effectively gone (e.g., less than 1% of initial fuel remaining)
        fuel_gone = False
        if self.initial_diagnostics and self.initial_diagnostics['total_fuel_remaining'] > 0:
            fuel_gone = diagnostics['total_fuel_remaining'] < 0.01 * self.initial_diagnostics['total_fuel_remaining']
        elif diagnostics['total_fuel_remaining'] == 0: # If initial fuel was 0 or very low, and now it's 0
             fuel_gone = True

        # Check if all agents are effectively out of all resources
        all_water_depleted = all(cw < self.action_costs['water'] for cw in self.current_water)
        all_retardant_depleted = all(cr < self.action_costs['retardant'] for cr in self.current_retardant)
        all_air_support_depleted = all(cas < self.action_costs['air_support'] for cas in self.current_air_support)
        
        agents_ineffective = all_water_depleted and all_retardant_depleted and all_air_support_depleted

        # Episode terminates if fire is out, fuel is gone, max steps reached, or agents are ineffective
        return (fire_out or fuel_gone or self.current_step >= self.max_steps or agents_ineffective)

    def _get_observation(self) -> np.ndarray:
        """
        Extract 2D observation from 3D simulation state using efficient vectorized operations.
        Returns a single global observation for the network.
        """

        obs = np.zeros((self.grid_size_x, self.grid_size_y, len(self.observation_channels)), dtype=np.float32)

        # Helper function for standardization to avoid repetitive code
        def standardize(data):
            mu = np.mean(data)
            sigma = np.std(data)
            if sigma > 0:
                return (data - mu) / sigma
            else:
                return data - mu

        # Heat Flux (standardized)
        obs_channel = self._downsample_channel(self.sim.heat_flux, self.target_shape, np.max)
        obs[:, :, 0] = standardize(obs_channel)

        # Energy Packets (standardized)
        obs_channel = self._downsample_channel(self.sim.EPs, self.target_shape, np.max)
        obs[:, :, 1] = standardize(obs_channel)

        # Temperature (standardized)
        obs_channel = self._downsample_channel(self.sim.temperature, self.target_shape, np.max)
        obs[:, :, 2] = standardize(obs_channel)
        
        # Fuel Density (standardized)
        obs_channel = self._downsample_channel(self.sim.fuel_density, self.target_shape, np.max)
        obs[:, :, 3] = standardize(obs_channel)
        
        # Moisture Fraction (standardized)
        obs_channel = self._downsample_channel(self.sim.moisture_fraction, self.target_shape, np.mean)
        obs[:, :, 4] = standardize(obs_channel)
        
        # Wind Magnitude (standardized)
        wind_field = self.sim.wind_manager.get_wind_field()
        wind_mag_3d = np.sqrt(wind_field['u']**2 + wind_field['v']**2 + wind_field['w']**2)
        obs_channel = self._downsample_channel(wind_mag_3d, self.target_shape, np.max)
        obs[:, :, 5] = standardize(obs_channel)

        # Agent Positions (binary mask)
        obs[:, :, 6] = 0.0
        for i in range(self.num_agents):
            agent_x, agent_y = self.agent_pos[i]
            if 0 <= agent_x < self.grid_size_x and 0 <= agent_y < self.grid_size_y:
                obs[agent_x, agent_y, 6] = 1.0

        # Burn Box Mask (binary mask)
        obs[:, :, 7] = self.burn_box_obs_mask.astype(np.float32)

        # Placeholder channel
        obs[:, :, 8] = 0.0
        
        return obs

    def _apply_action_at_target(self, agent_idx: int, action_name: str, x_grid: int, y_grid: int):
        """
        Apply firefighting action to simulation for a specific agent,
        managing that agent's resources, when agent is AT THE TARGET LOCATION OR WITHIN RANGE.
        """
        if action_name is None or action_name == 'idle':
            if self.verbose and action_name == 'idle':
                print(f"Agent {agent_idx}: Idle action, no changes applied.")
            return

        # Convert to simulation environment grid region (for the *target* cell or area)
        sim_x_start, sim_x_end, sim_y_start, sim_y_end = self._get_sim_coords(x_grid, y_grid)

        # Determine the appropriate z_max for the action
        terrain_z_index = self.sim._get_terrain_z_index(sim_x_start, sim_y_start)
        action_z_max = min(self.sim.nz - 1, terrain_z_index + self.action_z_ranges[action_name])

        if action_name == 'water':
            cost = self.action_costs['water']
            if self.current_water[agent_idx] >= cost:
                self.sim.apply_firefighting_action(action_name, sim_x_start, sim_y_start, sim_x_end, sim_y_end, z_max=action_z_max)
                self.current_water[agent_idx] -= cost
            else:
                if self.verbose: print(f"Agent {agent_idx} tried to water but insufficient resources.")
                self.agent_action_invalid_due_to_resources[agent_idx] = True
        elif action_name == 'firebreak':
            self.sim.apply_firefighting_action(action_name, sim_x_start, sim_y_start, sim_x_end, sim_y_end, z_max=action_z_max)
        elif action_name == 'retardant':
            cost = self.action_costs['retardant']
            if self.current_retardant[agent_idx] >= cost:
                self.sim.apply_firefighting_action(action_name, sim_x_start, sim_y_start, sim_x_end, sim_y_end, z_max=action_z_max)
                self.current_retardant[agent_idx] -= cost
            else:
                if self.verbose: print(f"Agent {agent_idx} tried to apply retardant but insufficient resources.")
                self.agent_action_invalid_due_to_resources[agent_idx] = True
        elif action_name == 'air_support':
            cost = self.action_costs['air_support']

            # Air support affects a radius around the target (x_grid, y_grid).
            air_x_min_grid = max(0, x_grid - QuicFireConstants.AIR_SUPPORT_GRID_RADIUS)
            air_y_min_grid = max(0, y_grid - QuicFireConstants.AIR_SUPPORT_GRID_RADIUS)
            air_x_max_grid = min(self.grid_size_x - 1, x_grid + QuicFireConstants.AIR_SUPPORT_GRID_RADIUS)
            air_y_max_grid = min(self.grid_size_y - 1, y_grid + QuicFireConstants.AIR_SUPPORT_GRID_RADIUS)

            sim_air_x_start = int(air_x_min_grid * (self.sim.nx / self.grid_size_x))
            sim_air_y_start = int(air_y_min_grid * (self.sim.ny / self.grid_size_y))
            sim_air_x_end = int((air_x_max_grid + 1) * (self.sim.nx / self.grid_size_x)) - 1
            sim_air_y_end = int((air_y_max_grid + 1) * (self.sim.ny / self.grid_size_y)) - 1

            sim_air_x_end = min(sim_air_x_end, self.sim.nx - 1)
            sim_air_y_end = min(sim_air_y_end, self.sim.ny - 1)
            
            # For air support, action_z_max will be None (all layers) as determined by _get_action_z_max
            if self.current_air_support[agent_idx] >= cost:
                self.sim.apply_firefighting_action(action_name, sim_air_x_start, sim_air_y_start, sim_air_x_end, sim_air_y_end, z_max=action_z_max)
                self.current_air_support[agent_idx] -= cost
            else:
                if self.verbose: print(f"Agent {agent_idx} tried to call air support but insufficient resources.")
                self.agent_action_invalid_due_to_resources[agent_idx] = True

    def _is_within_action_range(self, agent_idx: int, target_x: int, target_y: int, action_name: str) -> bool:
        """
        Checks if the agent is within the specified action range of the target.
        Uses Chebyshev distance (max of dx, dy) for square-like ranges.
        """
        current_agent_x, current_agent_y = self.agent_pos[agent_idx]
        
        dx = abs(current_agent_x - target_x)
        dy = abs(current_agent_y - target_y)

        if self.action_ranges[action_name]:
            return max(dx, dy) <= self.action_ranges[action_name]
        else:
            # Actions that don't require range checks (e.g. air_support) are always valid
            return False

    def _compute_global_reward(self, done: bool) -> float:
        """
        Compute reward for the entire environment, focusing on containment and minimizing outside burn.
        Heat flux values are in kW per cell.
        """
        reward = 0.0
        
        # --- PRIMARY GOAL: Minimize Fire and Fuel Consumption Outside Burn Box ---

        outside_burn_box_mask_sim_2d = ~self.burn_box_mask_sim

        # 1. Penalty for Active Fire (Heat Flux) outside Burn Box
        current_heat_flux_per_column_2d = np.sum(self.sim.heat_flux, axis=-1)
        current_heat_flux_outside_bb = np.sum(current_heat_flux_per_column_2d[outside_burn_box_mask_sim_2d])

        num_cells_outside_bb_2d = np.sum(outside_burn_box_mask_sim_2d)
        max_possible_total_heat_flux_outside_bb = self.max_heat_flux_per_cell * num_cells_outside_bb_2d

        active_fire_penalty = 0.0
        if max_possible_total_heat_flux_outside_bb > 0:
            normalized_heat_flux_outside_bb = current_heat_flux_outside_bb / max_possible_total_heat_flux_outside_bb
            active_fire_penalty = normalized_heat_flux_outside_bb * QuicFireConstants.PENALTY_ACTIVE_FIRE_OUTSIDE_BB
            reward += active_fire_penalty
        
        if self.enable_reward_logging:
            self.global_reward_components_current_step['active_fire_outside_bb_penalty'] = active_fire_penalty

        # 2. Penalty for Fuel Consumed outside Burn Box (Historical Damage)
        pre_fuel_per_column_2d = np.sum(self._pre_action_fuel_density, axis=-1)
        current_fuel_per_column_2d = np.sum(self.sim.fuel_density, axis=-1)

        fuel_consumed_per_column_this_step_2d = pre_fuel_per_column_2d - current_fuel_per_column_2d
        
        fuel_consumed_outside_bb_this_step = np.sum(
            np.maximum(0, fuel_consumed_per_column_this_step_2d)[outside_burn_box_mask_sim_2d]
        )

        initial_total_fuel_outside_bb_at_start_of_episode = self.initial_fuel_outside_bb if self.initial_fuel_outside_bb > 0 else 1e-6

        fuel_consumed_penalty = 0.0
        if fuel_consumed_outside_bb_this_step > 0:
            normalized_fuel_consumed_outside_bb_this_step = fuel_consumed_outside_bb_this_step / initial_total_fuel_outside_bb_at_start_of_episode
            fuel_consumed_penalty = normalized_fuel_consumed_outside_bb_this_step * QuicFireConstants.PENALTY_FUEL_CONSUMED_OUTSIDE_BB
            reward += fuel_consumed_penalty

        if self.enable_reward_logging:
            self.global_reward_components_current_step['fuel_consumed_outside_bb_penalty'] = fuel_consumed_penalty

        # 3. Reward for Fuel Consumed Inside Burn Box (Maximizing the Burn)
        fuel_consumed_inside_bb_this_step = np.sum(
            np.maximum(0, fuel_consumed_per_column_this_step_2d)[self.burn_box_mask_sim]
        )

        initial_total_fuel_inside_bb_at_start_of_episode = self.initial_fuel_inside_bb if self.initial_fuel_inside_bb > 0 else 1e-6

        fuel_consumed_reward = 0.0
        if fuel_consumed_inside_bb_this_step > 0:
            normalized_fuel_consumed_inside_bb_this_step = fuel_consumed_inside_bb_this_step / initial_total_fuel_inside_bb_at_start_of_episode
            fuel_consumed_reward = normalized_fuel_consumed_inside_bb_this_step * QuicFireConstants.REWARD_FUEL_CONSUMED_INSIDE_BB
            reward += fuel_consumed_reward

        if self.enable_reward_logging:
            self.global_reward_components_current_step['fuel_consumed_inside_bb_reward'] = fuel_consumed_reward
        
        # --- Secondary Goals (Resource Efficiency, Time) ---

        # 3: Efficiency (Resource Usage Penalty)
        total_water_capacity_all = self.resource_capacity['water'] * self.num_agents
        total_retardant_capacity_all = self.resource_capacity['retardant'] * self.num_agents
        total_air_support_capacity_all = self.resource_capacity['air_support'] * self.num_agents

        total_water_used_episode = total_water_capacity_all - sum(self.current_water)
        total_retardant_used_episode = total_retardant_capacity_all - sum(self.current_retardant)
        total_air_support_used_episode = total_air_support_capacity_all - sum(self.current_air_support)

        water_penalty = 0.0
        if total_water_capacity_all > 0:
            water_penalty = QuicFireConstants.PENALTY_GLOBAL_WATER_USAGE * (total_water_used_episode / total_water_capacity_all)
            reward += water_penalty
        if self.enable_reward_logging: self.global_reward_components_current_step['water_usage_penalty'] = water_penalty

        retardant_penalty = 0.0
        if total_retardant_capacity_all > 0:
            retardant_penalty = QuicFireConstants.PENALTY_GLOBAL_RETARDANT_USAGE * (total_retardant_used_episode / total_retardant_capacity_all)
            reward += retardant_penalty
        if self.enable_reward_logging: self.global_reward_components_current_step['retardant_usage_penalty'] = retardant_penalty

        air_support_penalty = 0.0
        if total_air_support_capacity_all > 0:
            air_support_penalty = QuicFireConstants.PENALTY_GLOBAL_AIR_SUPPORT_USAGE * (total_air_support_used_episode / total_air_support_capacity_all)
            reward += air_support_penalty
        if self.enable_reward_logging: self.global_reward_components_current_step['air_support_usage_penalty'] = air_support_penalty

        # 4: Time Pressure (Penalty per step)
        time_penalty = QuicFireConstants.PENALTY_TIME_STEP 
        reward += time_penalty
        if self.enable_reward_logging:
            self.global_reward_components_current_step['time_penalty'] = time_penalty

        # Terminal Reward Considerations
        if done:
            if self.current_step >= self.max_steps:
                mission_failure_timeout_penalty = QuicFireConstants.PENALTY_MISSION_FAILURE_TIMEOUT
                reward += mission_failure_timeout_penalty
                if self.enable_reward_logging:
                    self.global_reward_components_current_step['mission_failure_timeout_penalty'] = mission_failure_timeout_penalty
            elif self._check_win_condition():
                mission_success_reward = QuicFireConstants.REWARD_MISSION_CONTAINED_SUCCESS
                reward += mission_success_reward
                if self.enable_reward_logging:
                    self.global_reward_components_current_step['mission_success_reward'] = mission_success_reward
            else:
                mission_failure_spread_penalty = QuicFireConstants.PENALTY_MISSION_FAILURE_SPREAD
                reward += mission_failure_spread_penalty
                if self.enable_reward_logging:
                    self.global_reward_components_current_step['mission_failure_spread_penalty'] = mission_failure_spread_penalty
        
        # Clip the final reward
        reward = np.clip(reward, -QuicFireConstants.MAX_GLOBAL_REWARD_CLIP, QuicFireConstants.MAX_GLOBAL_REWARD_CLIP)
        
        if self.enable_reward_logging:
            self.global_reward_components_current_step['total_reward'] = reward

        return reward
    
    def _calculate_per_agent_rewards(self, intents_before: List[Dict]) -> List[float]:
        """
        Calculates the per-agent rewards for the current environment step.
        """
        per_agent_rewards = [0.0] * self.num_agents

        for i in range(self.num_agents):
            
            if self.enable_reward_logging:
                agent_components = self.per_agent_reward_components_current_step[i]

            # Only calculate reward if the agent had an active intent or just terminated its option
            if intents_before[i] is not None or self.option_terminated[i]:
                # Agent-specific reward based on action type
                if self.option_terminated[i] and intents_before[i] is not None:
                    action_type = intents_before[i]['type']
                    x, y = intents_before[i]['x'], intents_before[i]['y']
                    
                    # Apply reward for successful action at the target cell
                    if action_type in [self.action_type_map['water'], 
                                       self.action_type_map['retardant'], 
                                       self.action_type_map['air_support']]:
                        fire_reduction_percentage = self._check_fire_reduction(x, y) # This uses heat_flux
                        fire_reduction_reward = fire_reduction_percentage * QuicFireConstants.REWARD_AGENT_FIRE_REDUCTION
                        per_agent_rewards[i] += fire_reduction_reward
                        if self.enable_reward_logging:
                            agent_components['fire_reduction'] = fire_reduction_reward # Assign directly

                # NOTE: Firebreaks are not rewarded because they reduce vegetation but prevent larger fires making them neutral.
                # Agent should learn to minimally use firebreaks to reduce larger fires, not be rewarded for reducing vegetation.

                # Apply movement penalty
                movement_cost_value = self.last_movement_cost_per_agent[i]
                per_agent_rewards[i] += movement_cost_value
                if self.enable_reward_logging:
                    agent_components['movement_cost'] = movement_cost_value # Assign directly

                # Apply penalty for invalid actions due to insufficient resources
                if self.agent_action_invalid_due_to_resources[i]:
                    invalid_action_penalty = QuicFireConstants.PENALTY_RESOURCE_INVALID_ACTION
                    per_agent_rewards[i] += invalid_action_penalty
                    if self.enable_reward_logging:
                        agent_components['invalid_action_penalty'] = invalid_action_penalty
            
            # Clip per-agent rewards
            per_agent_rewards[i] = np.clip(per_agent_rewards[i], -QuicFireConstants.MAX_AGENT_REWARD_CLIP, QuicFireConstants.MAX_AGENT_REWARD_CLIP)

            if self.enable_reward_logging:
                agent_components['total_reward'] = per_agent_rewards[i] # Update total after clipping

        return per_agent_rewards
    
    def _check_fire_reduction(self, x_grid: int, y_grid: int) -> float:
        """
        Calculates the reduction in total heat flux in the simulation region corresponding to the given
        agent grid coordinates, normalized by initial heat flux.
        Returns a value between 0.0 and 1.0 (or higher if fire intensity could go below 0 which is not the case).
        """
        if self._pre_action_heat_flux is None:
            return 0.0

        sim_x_start, sim_x_end, sim_y_start, sim_y_end = self._get_sim_coords(x_grid, y_grid)

        # Extract relevant slices for pre- and post-action heat flux across ALL z-layers
        pre_action_heat_flux_region = self._pre_action_heat_flux[sim_x_start:sim_x_end+1, sim_y_start:sim_y_end+1, :]
        current_heat_flux_region = self.sim.heat_flux[sim_x_start:sim_x_end+1, sim_y_start:sim_y_end+1, :]

        pre_fire_intensity_sum = np.sum(pre_action_heat_flux_region)
        current_fire_intensity_sum = np.sum(current_heat_flux_region)

        # Calculate the reduction in fire intensity (heat flux).
        fire_reduction = pre_fire_intensity_sum - current_fire_intensity_sum
        
        if pre_fire_intensity_sum > 0:
            # Reward is proportional to the fraction of heat flux reduced in that region.
            # Max(0.0, ...) ensures no negative reward if heat flux unexpectedly increased (shouldn't happen with direct zeroing).
            return max(0.0, fire_reduction / pre_fire_intensity_sum)
        else:
            # If there was no heat flux initially in this region, no fire to extinguish, reward is 0.
            return 0.0
    
    def _downsample_channel(self, data_3d: np.ndarray, target_shape_2d: Tuple[int, int], func) -> np.ndarray:
        """
        Helper to downsample a 3D simulation channel to a 2D observation channel.
        First reduces along z-axis, then block-reduces to target_shape_2d.

        Args:
            data_3d (np.ndarray): The 3D data array from the simulation.
            target_shape_2d (Tuple[int, int]): The desired (x, y) dimensions for the 2D output.
            func: The aggregation function to apply (e.g., np.max, np.mean).

        Returns:
            np.ndarray: The downsampled 2D array.
        """
        # Reduce along z-axis first
        data_2d = func(data_3d, axis=2)

        # Calculate block sizes for downsampling
        block_x = data_2d.shape[0] // target_shape_2d[0]
        block_y = data_2d.shape[1] // target_shape_2d[1]

        # Use block_reduce for efficient downsampling
        downsampled = block_reduce(data_2d, block_size=(block_x, block_y), func=func, cval=0)
        
        # Ensure the output matches the target shape exactly by slicing
        return downsampled[:target_shape_2d[0], :target_shape_2d[1]]
    
    def _get_sim_coords(self, grid_x: int, grid_y: int) -> Tuple[int, int, int, int]:
        """
        Helper to convert agent's coarse grid coordinates to simulation's fine grid coordinates.
        Returns (sim_x_start, sim_x_end, sim_y_start, sim_y_end) representing the region
        in the simulation grid corresponding to the given agent grid cell.
        """
        # Calculate scaling factors
        scale_x = self.sim.nx / self.grid_size_x
        scale_y = self.sim.ny / self.grid_size_y

        sim_x_start = int(grid_x * scale_x)
        sim_y_start = int(grid_y * scale_y)

        # The end coordinates should cover up to, but not including, the next grid cell's start
        # Ensure we don't go out of bounds of the simulation grid
        sim_x_end = min(int((grid_x + 1) * scale_x) - 1, self.sim.nx - 1)
        sim_y_end = min(int((grid_y + 1) * scale_y) - 1, self.sim.ny - 1)
        
        return sim_x_start, sim_x_end, sim_y_start, sim_y_end
    
    def _initialize_agent_positions(self, num_agents: int, max_attempts_per_agent: int = 1000, random_seed: int = None) -> List[Tuple[int, int]]:
        """
        Initializes agent positions ensuring they are outside the burn box.
        Attempts to find unique positions for each agent.
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        agent_positions = []
        occupied_cells = set() # To ensure unique starting positions if desired

        grid_x, grid_y = self.grid_size_x, self.grid_size_y
        
        for i in range(num_agents):
            found_position = False
            for attempt in range(max_attempts_per_agent):
                # Randomly pick a coordinate within the observation grid
                x = random.randint(0, grid_x - 1)
                y = random.randint(0, grid_y - 1)
                
                # Check if the chosen cell is outside the burn box
                # self.burn_box_obs_mask is True for cells *inside* the burn box
                is_outside_burn_box = not self.burn_box_obs_mask[x, y]
                
                # Check if the cell is already occupied by another agent
                is_occupied = (x, y) in occupied_cells

                if is_outside_burn_box and not is_occupied:
                    agent_positions.append((x, y))
                    occupied_cells.add((x, y))
                    found_position = True
                    break
            
            if not found_position:
                raise RuntimeError(f"Could not find a valid starting position for agent {i} after {max_attempts_per_agent} attempts. "
                                   "Consider increasing grid size, reducing burn box size, or increasing attempts.")
        
        return agent_positions
    
    def _check_win_condition(self) -> bool:
        """
        Determines if the agent has "won" the current episode based on the criterion
        that no significant fuel was burned outside the burn box.
        Assumes fire is already extinguished when this function is called.
        """

        # Project 3D fuel densities to 2D total fuel per column for masked sums
        current_fuel_per_column_2d = np.sum(self.sim.fuel_density, axis=-1)

        # Pre-calculate masks (assuming self.burn_box_mask_sim is 2D)
        outside_burn_box_mask_2d = ~self.burn_box_mask_sim

        # --- Win Condition: Minimal Fuel Burned *Outside* the Burn Box ---
        initial_fuel_outside_bb = np.sum(self.initial_fuel_per_column_2d[outside_burn_box_mask_2d])
        current_fuel_outside_bb = np.sum(current_fuel_per_column_2d[outside_burn_box_mask_2d])
        percent_fuel_burned_outside_bb = (initial_fuel_outside_bb - current_fuel_outside_bb) / initial_fuel_outside_bb

        # Check if fuel burned outside is below a very small threshold
        if percent_fuel_burned_outside_bb < QuicFireConstants.WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD:
            return True # Too much fuel burned outside the burn box, so not a win

        # If this condition passes, the agent has won
        return False

    def close(self):
        """
        Cleans up resources held by the environment.
        Specifically, it ensures the QuicFireEngine simulation is properly closed.
        """
        if hasattr(self, 'sim') and self.sim is not None:
            if hasattr(self.sim, 'close') and callable(self.sim.close):
                print("Closing QuicFireEngine simulation.")
                self.sim.close()
            else:
                print("No explicit close/destroy method found for QuicFireEngine, relying on garbage collection.")
            self.sim = None # Set to None to indicate it's no longer active

        print("FirefighterEnvironment closed.")

    def _compute_fire_gradient(self, prev_fire_grid: np.ndarray, curr_fire_grid: np.ndarray) -> np.ndarray:
        return curr_fire_grid - prev_fire_grid

    def _calculate_reachable_cells(self, agent_pos: Tuple[int, int], target_mask: np.ndarray, time_step: int, action_range: float) -> np.ndarray:
        agent_x, agent_y = agent_pos
        x, y = np.indices((self.grid_size_x, self.grid_size_y))
        distances = np.abs(x - agent_x) + np.abs(y - agent_y)
        
        # Use a strict == criteria, as requested
        reachable_mask = (distances == (time_step + action_range)) & target_mask
        return reachable_mask

    # Assuming this is your helper function within the class
    def _predict_fire_spread(self, curr_fire_grid: np.ndarray, gradient: np.ndarray, rollout_steps: int = 20) -> List[np.ndarray]:
        """
        Predict fire spread for rollout_steps using only the fire gradient.
        This is a computationally cheap heuristic, as requested.
        """
        predicted_grids = [curr_fire_grid.copy()]
        current_grid = curr_fire_grid.copy()
        
        for _ in range(rollout_steps):
            # Fire spreads based on its existing gradient
            new_grid = np.clip(current_grid + gradient, 0, 1)
            predicted_grids.append(new_grid)
            current_grid = new_grid
        
        return predicted_grids

    # --- The main refactored function ---
    def get_valid_actions_mask(self, agent_idx: int) -> Dict[str, np.ndarray]:
        """
        Generates a mask for valid action types and joint (x,y) coordinates for a given agent
        using a cheap rollout prediction.
        """
        rollout_steps = 20
        
        # Initialize masks
        action_type_mask = np.ones(self.num_action_types, dtype=bool)
        target_xy_mask_2d = np.zeros(
            (self.num_action_types, self.grid_size_x, self.grid_size_y), dtype=bool
        )

        # Get agent-specific info
        agent_pos = self.agent_pos[agent_idx]
        current_agent_x, current_agent_y = agent_pos

        # Map action types to their indices
        idx_idle = self.action_type_map['idle']
        idx_water = self.action_type_map['water']
        idx_firebreak = self.action_type_map['firebreak']
        idx_retardant = self.action_type_map['retardant']
        idx_air_support = self.action_type_map['air_support']

        # Resource checks: This part is always valid and does not depend on fire.
        if self.current_water[agent_idx] < self.action_costs['water']:
            action_type_mask[idx_water] = False
        if self.current_retardant[agent_idx] < self.action_costs['retardant']:
            action_type_mask[idx_retardant] = False
        if self.current_air_support[agent_idx] < self.action_costs['air_support']:
            action_type_mask[idx_air_support] = False
        # Firebreak is always available (no resource cost)

        # Extract fuel density and burn box perimeter (always needed)
        fuel_density_obs = self._downsample_channel(self.sim.fuel_density, self.target_shape, np.max)
        fuel_mask = fuel_density_obs > QuicFireConstants.FUEL_DENSITY_THRESHOLD
        dilated_burn_box = binary_dilation(self.burn_box_obs_mask, structure=np.ones((3,3), dtype=bool))
        burn_box_perimeter = dilated_burn_box & ~self.burn_box_obs_mask

        # --- NEW: Check for initial simulation step with no fire ---
        if self._pre_action_agent_obs_heat_flux is None:
            # In this case, only Firebreaks and Idle are valid actions
            
            # Firebreaks are only valid on the burn box perimeter with fuel.
            # All such locations are considered valid targets.
            firebreak_targets = burn_box_perimeter & fuel_mask
            target_xy_mask_2d[idx_firebreak, :, :] = firebreak_targets
            
            # Idle is always a valid action at the agent's current position
            target_xy_mask_2d[idx_idle, current_agent_x, current_agent_y] = True

            # All other actions are invalid as there is no fire to apply them to
            action_type_mask[idx_water] = False
            action_type_mask[idx_retardant] = False
            action_type_mask[idx_air_support] = False
            # The firebreak action type is only valid if there are any firebreak targets
            action_type_mask[idx_firebreak] = action_type_mask[idx_firebreak] and np.any(firebreak_targets)
            
            return {
                'action_type_mask': action_type_mask,
                'target_xy_mask_2d': target_xy_mask_2d
            }

        # --- If there IS fire, proceed with the full rollout prediction ---
        prev_fire_grid = self._pre_action_agent_obs_heat_flux
        curr_fire_grid = self._current_action_agent_obs_heat_flux
        
        # Compute fire gradient and predict fire spread for the rollout
        gradient = self._compute_fire_gradient(prev_fire_grid, curr_fire_grid)
        predicted_grids = self._predict_fire_spread(curr_fire_grid, gradient, rollout_steps)

        # Combine masks across all rollout steps
        combined_masks = {action: np.zeros((self.grid_size_x, self.grid_size_y), dtype=bool) for action in self.action_type_map}

        for t in range(1, rollout_steps + 1):
            predicted_grid = predicted_grids[t]
            
            buffered_fire_grid = np.zeros_like(predicted_grid, dtype=bool)
            if np.any(predicted_grid > 0):
                buffered_fire_grid = binary_dilation(predicted_grid > 0, structure=np.ones((3,3), dtype=bool))
            
            valid_target_cells = ~self.burn_box_obs_mask

            firebreak_targets = (burn_box_perimeter & fuel_mask) | (buffered_fire_grid & valid_target_cells)
            firebreak_mask = self._calculate_reachable_cells(agent_pos, firebreak_targets, t, self.action_ranges['firebreak'])
            combined_masks['firebreak'] |= firebreak_mask
            
            for action in ['water', 'retardant', 'air_support']:
                fire_targets = buffered_fire_grid & valid_target_cells
                action_mask = self._calculate_reachable_cells(agent_pos, fire_targets, t, self.action_ranges[action])
                combined_masks[action] |= action_mask

        for action, action_idx in self.action_type_map.items():
            if action == 'idle':
                idle_mask = np.zeros((self.grid_size_x, self.grid_size_y), dtype=bool)
                if 0 <= current_agent_x < self.grid_size_x and 0 <= current_agent_y < self.grid_size_y:
                    idle_mask[current_agent_x, current_agent_y] = True
                target_xy_mask_2d[action_idx, :, :] = idle_mask
            
            else:
                if action_type_mask[action_idx]:
                    target_xy_mask_2d[action_idx, :, :] = combined_masks[action]
                else:
                    target_xy_mask_2d[action_idx, :, :] = False

        return {
            'action_type_mask': action_type_mask,
            'target_xy_mask_2d': target_xy_mask_2d
        }