import numpy as np
import time
from scipy.ndimage import convolve
from typing import Dict, Optional, Tuple, List, Literal, Any
from QuicFire.QuicFireBurnBox import QuicFireBurnBox
from QuicFire.QuicFireIgnitionManager import QuicFireIgnitionManager
from QuicFire.QuicFireWindFieldManager import QuicFireWindFieldManager
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireEnvGenerator import VegetationType, VEGETATION_PROPERTIES
from QuicFire.QuicFireTuning import apply_tuning
from QuicFire.RNGCollection import RNGCollection
from scipy.ndimage import label # Import for connected components

class QuicFireEngine:
    def __init__(self, config: Dict, veg_map: np.ndarray, fuel_map: np.ndarray, fuel_moisture_map: np.ndarray, height_map: np.ndarray, burn_box: Optional[QuicFireBurnBox]=None, wind_updates: Optional[List[Dict]]=None, ignition_lines: Optional[List[Tuple[float, float, float, float]]]=None, module_seeds: Optional[RNGCollection]=None, tuning: Optional[Dict]=None):
        """
        Initializes the QuicFire simulation engine with a given configuration.
        """
        if tuning:
            apply_tuning(tuning)
        self.config = config if config is not None else {}
        self._validate_config_file()

        # Core simulation parameters
        self.total_steps = 0
        self.current_simulation_time = 0.0
        self.nx = self.config.get('nx', QuicFireConstants.DEFAULT_NX)
        self.ny = self.config.get('ny', QuicFireConstants.DEFAULT_NY)
        self.nz = self.config.get('nz', QuicFireConstants.DEFAULT_NZ)
        self.dx = self.config.get('dx', QuicFireConstants.DEFAULT_DX)
        self.dy = self.config.get('dy', QuicFireConstants.DEFAULT_DY)
        self.max_grid_height = self.config.get('max_grid_height', QuicFireConstants.DEFAULT_MAX_GRID_HEIGHT)
        self.total_time = self.config.get('total_time', QuicFireConstants.DEFUALT_SIMULATION_TIME)
        
        # Initialize Random Number Generators
        self._setup_rng_generators(module_seeds=module_seeds)

        # Simulation state and grid initialization
        self.z_levels, self.dz = self._create_vertical_grid()
        self._initialize_state()
        self._validate_parameters()

        # Initialize core simulation components
        self._setup_terrain_and_vegetation(veg_map, fuel_map, fuel_moisture_map, height_map)
        self._setup_wind_manager(wind_updates)
        self._setup_burn_box(burn_box)
        self._setup_ignition(ignition_lines)

        # Engine closed flag
        self._is_closed = False

        # DEBUGGING:
        self.save_interval = None
        self.consumption_rate_history = []

        print(f"✓ Fuel distribution set: Total fuel {np.sum(self.fuel_density):.1f} kg")

    def _setup_rng_generators(self, module_seeds: Optional[RNGCollection]=None):
        if module_seeds:
            self.module_seeds = module_seeds
        else:
            self.module_seeds = RNGCollection(self.config.get('random_seed', QuicFireConstants.DEFAULT_RANDOM_SEED))

        self.rng = self.module_seeds.get_rng('master')

    def _setup_terrain_and_vegetation(self, veg_map: np.ndarray, fuel_map: np.ndarray, fuel_moisture_map: np.ndarray, height_map: np.ndarray) -> None:
        # Check that inputs are NumPy arrays
        if not all(isinstance(arr, np.ndarray) for arr in [veg_map, fuel_map, fuel_moisture_map, height_map]):
            raise ValueError("veg_map, fuel_map, and height_map must be NumPy arrays")

        # Check dimensions match (nx, ny)
        expected_shape = (self.nx, self.ny)
        for arr, name in [(veg_map, "veg_map"), (fuel_map, "fuel_map"), (fuel_moisture_map, "fuel_moisture_map"), (height_map, "height_map")]:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} shape {arr.shape} does not match expected shape {expected_shape}")
            
        self.height_map = height_map
        self.fuel_density = self._generate_3d_fuel_grid(veg_map=veg_map, fuel_map=fuel_map, height_map=height_map)
        self.moisture_fraction = self._generate_3d_moisture_grid(veg_map=veg_map, fuel_moisture_map=fuel_moisture_map, height_map=height_map)
        self.initial_fuel = np.sum(self.fuel_density)

    def _generate_3d_fuel_grid(self,
            veg_map: np.ndarray,
            fuel_map: np.ndarray,
            height_map: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D fuel density grid with shape (nx, ny, nz)"""
        fuel_density = np.zeros((self.nx, self.ny, self.nz))

        for x in range(self.nx):
            for y in range(self.ny):

                veg_type = VegetationType(veg_map[x, y])
                properties = VEGETATION_PROPERTIES[veg_type]
                surface_fuel_load = fuel_map[x, y]  # kg/m²
                current_terrain_height = height_map[x, y]
                veg_height = properties.height
                veg_max_height = current_terrain_height + veg_height
                
                # Find the lowest z-grid index (k_ground) that is at or above the terrain height
                k_ground = self._get_terrain_z_index(x, y)

                # Get dz for the current k_ground cell or an average dz
                dz_at_ground = self.dz
                if not np.isscalar(self.dz) and k_ground < len(self.z_levels) - 1:
                     dz_at_ground = self.z_levels[k_ground+1] - self.z_levels[k_ground]
                elif not np.isscalar(self.dz) and k_ground == len(self.z_levels) - 1 and k_ground > 0:
                     dz_at_ground = self.z_levels[k_ground] - self.z_levels[k_ground-1] # Use thickness of previous layer as fallback

                # --- Add Surface Fuel Density ---
                if surface_fuel_load is not None and k_ground < self.nz:
                    # Distribute surface density over a small fixed height, e.g., 0.1m or the cell dz
                    # Convert kg/m^2 to kg/m^3. Use a small, fixed effective height for surface fuel or dz_at_ground
                    effective_surface_fuel_height = min(0.2, dz_at_ground) # Assume surface fuels occupy up to 20cm or cell dz
                    
                    volumetric_surface_density = surface_fuel_load / max(0.001, effective_surface_fuel_height)
                    
                    fuel_density[x, y, k_ground] += volumetric_surface_density

                # --- Add Elevated Fuel Density ---
                for k in range(k_ground, self.nz): 
                    current_z_height = self.z_levels[k]
                    relative_height = current_z_height - current_terrain_height
                    
                    if relative_height < 0: 
                        continue
                    
                    # Exponential Vertical density profile
                    density_from_profile = properties.fuel_load * np.exp(-relative_height / properties.height) if relative_height <= properties.height else 0.0
                    if density_from_profile > 0.001: # Small threshold to add fuel
                        fuel_density[x, y, k] += density_from_profile
                    
                    if relative_height > veg_max_height * 1.5: 
                         break
        
        fuel_density = np.clip(fuel_density, 0.0, QuicFireConstants.RHO_F0) # Cap for realism
        
        return fuel_density
    
    def _generate_3d_moisture_grid(self,
            veg_map: np.ndarray,
            fuel_moisture_map: np.ndarray,
            height_map: np.ndarray,
        ) -> np.ndarray:
        """
        Generate 3D moisture fraction grid with shape (nx, ny, nz).
        """
        # Initialize the 3D moisture grid with zeros
        moisture_grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

        # Iterate over each (x, y) column in the 2D grid
        for x in range(self.nx):
            for y in range(self.ny):
                # Get vegetation type and its properties for the current (x, y) location
                veg_type = VegetationType(veg_map[x, y])
                properties = VEGETATION_PROPERTIES[veg_type]
                
                # Get the base moisture fraction from the 2D moisture map
                base_moisture_fraction = fuel_moisture_map[x, y] # This is 0-1
                
                # Get terrain height and vegetation height
                current_terrain_height = height_map[x, y]
                veg_height = properties.height # Height of the vegetation type
                
                # Calculate the absolute maximum height of vegetation above ground
                veg_max_height_absolute = current_terrain_height + veg_height
                
                # Find the lowest z-grid index (k_ground) that is at or above the terrain height
                k_ground = self._get_terrain_z_index(x, y)

                # --- Assign Moisture to Ground Level (k_ground) ---
                # For water vegetation type, ensure it's fully saturated at ground level
                if veg_type == VegetationType.WATER and k_ground < self.nz:
                    moisture_grid[x, y, k_ground] = 1.0 # Water cells are fully saturated
                elif k_ground < self.nz:
                    # For other vegetation, assign the base moisture fraction to the ground cell
                    moisture_grid[x, y, k_ground] = base_moisture_fraction

                # --- Distribute Moisture Vertically for Elevated Layers ---
                for k in range(k_ground, self.nz): 
                    current_z_height = self.z_levels[k]
                    
                    # Calculate height relative to the terrain surface
                    relative_height = current_z_height - current_terrain_height
                    
                    # If the current z-level is below the terrain, skip it
                    if relative_height < 0: 
                        continue 
                    
                    # The decay factor is based on relative height and vegetation height.
                    if properties.height > 0:
                        decay_factor = np.exp(-relative_height / properties.height)
                    else:
                        # For vegetation types with zero height (e.g., BARE_SOIL, WATER),
                        decay_factor = 0.0 
                    
                    # Calculate moisture fraction for the current cell
                    # This is the base moisture fraction modulated by the vertical decay.
                    moisture_at_level = base_moisture_fraction * decay_factor

                    moisture_grid[x, y, k] = moisture_at_level
                    
                    if relative_height > veg_max_height_absolute * 1.5: 
                         break # Stop processing higher z-levels for this column

        moisture_grid = np.clip(moisture_grid, 0.0, 1.0)
        
        return moisture_grid

    def _setup_wind_manager(self, wind_updates: Optional[List[Dict]]=None) -> None:
        """Initializes the wind field manager and generates wind updates."""
        self.wind_manager = QuicFireWindFieldManager(
            self.module_seeds.get_rng('wind'), self.nx, self.ny, self.nz, self.z_levels, self.dx, self.dy, self.fuel_density, self.height_map
        )
        
        if wind_updates:
            self.wind_updates = wind_updates
        else:
            self.wind_updates = self.wind_manager.generate_rand_wind_updates(QuicFireConstants.NUM_WIND_UPDATES, self.total_time)
        
        # DEBUGGING:
        # self.wind_manager.visualize_wind_field()
        # self.wind_manager.visualize_2d_slices(slice_type='xy', slice_idx=4, 
        #                                   subsample_factor=2, plot_terrain_contour=True,
        #                                   title="Wind Field at Z=2m")

    def _setup_burn_box(self, burn_box: Optional[QuicFireBurnBox]=None) -> None:
        """Initializes the burn box coordinates."""
        if burn_box:
            self.burn_box = burn_box
            self.burn_box_coords = self.burn_box.get_coords()
        else:
            self.burn_box = QuicFireBurnBox(
                rng=self.module_seeds.get_rng('burn_box'),
                nx=self.nx, 
                ny=self.ny,
                dx=self.dx,
                dy=self.dy,
                ignition_inner_buffer=QuicFireConstants.IGNITION_INNER_BUFFER
            )
            self.burn_box_coords = self.burn_box.get_coords()

    def _setup_ignition(self, ignition_lines: Optional[List[Tuple[float, float, float, float]]]=None) -> None:
        """Sets up the ignition lines and applies initial ignition."""
        self.ignition_manager = QuicFireIgnitionManager(self.module_seeds.get_rng('ignition'))
        self.ignition_manager.clear_lines()
        
        if not ignition_lines:
            assert self.burn_box_coords, "Missing required burn_box_coords for ignition lines"
            ignition_lines = self.ignition_manager.generate_rand_ignition_lines(
                nx=self.nx, 
                ny=self.ny, 
                dx=self.dx,
                dy=self.dy,
                num_lines=QuicFireConstants.NUM_IGNITION_LINES, 
                burn_box_coords=self.burn_box_coords,
                inner_buffer=QuicFireConstants.IGNITION_INNER_BUFFER
            )

        ignition_parameters = self.ignition_manager.generate_rand_ignition_params()
        self.ignition_manager.add_lines_from_coords(ignition_lines, **ignition_parameters)
        ignited_cells = self.ignition_manager.apply_ignition(self)
        print(f"✓ Ignited {ignited_cells} cells")

    def _validate_config_file(self) -> None:
        """Validates the presence of essential configuration parameters."""
        required_params = ['nx', 'ny', 'nz', 'dx', 'dy', 'total_time']
        for param in required_params:
            assert param in self.config, f"Missing required configuration parameter: '{param}'"

    def _validate_parameters(self) -> None:
        """Validates physical parameters and array shapes."""
        assert QuicFireConstants.DT > 0, "Time step must be positive"
        assert QuicFireConstants.T_BURNOUT > 0, "Burnout time must be positive"
        assert QuicFireConstants.C_IGSC > 0, "Critical ignition surface concentration must be positive"
        assert 0 <= QuicFireConstants.DEFAULT_MOISTURE_INIT <= 1, "Initial moisture must be between 0 and 1"
        assert QuicFireConstants.U_REF > 0, "Reference wind speed must be positive"
        assert QuicFireConstants.NUM_IGNITION_LINES > 0, "Number of ignition lines must be >= 1"
        assert QuicFireConstants.NUM_WIND_UPDATES > 0, "Number of wind updates must be >= 1"
        assert self.heat_flux.shape == (self.nx, self.ny, self.nz), "Heat flux shape mismatch"

    def _initialize_state(self) -> None:
        """Initializes state arrays with realistic fuel distributions."""
        self.moisture_fraction = np.full((self.nx, self.ny, self.nz), QuicFireConstants.DEFAULT_MOISTURE_INIT)
        self.fuel_density = np.zeros((self.nx, self.ny, self.nz))
        self.EPs = np.zeros((self.nx, self.ny, self.nz))
        self.active_fire_time = np.zeros((self.nx, self.ny, self.nz))
        self.moisture_depletion_fraction = np.zeros((self.nx, self.ny, self.nz))
        self.temperature = np.ones((self.nx, self.ny, self.nz)) * QuicFireConstants.T_AMBIENT
        self.heat_flux = np.zeros((self.nx, self.ny, self.nz))
        self.consumption = {'surface': 0.0, 'midstory': 0.0, 'canopy': 0.0}

    def _create_vertical_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a stretched vertical grid with adaptive resolution."""
        surface_height = next(v for k, v in QuicFireConstants.SURFACE_HEIGHTS.items() if self.nz >= k)
        surface_spacing = next(v for k, v in QuicFireConstants.SURFACE_SPACINGS.items() if self.nz >= k)
        n_surface = min(int(surface_height // surface_spacing) + 1, self.nz - QuicFireConstants.MIN_SURFACE_POINTS)
        z_surface = np.linspace(0, surface_height, n_surface)
        n_upper = self.nz - len(z_surface)
        
        if n_upper > 0:
            z_upper = np.linspace(surface_height, self.max_grid_height, n_upper + 1)[1:]
            z_levels = np.concatenate([z_surface, z_upper])
        else:
            z_levels = np.linspace(0, self.max_grid_height, self.nz)

        dz = np.diff(z_levels, prepend=z_levels[0])
        return z_levels, dz

    def _create_movement_kernel(self, wind_u: float = 0.0, wind_v: float = 0.0, wind_w: float = 0.0) -> np.ndarray:
        """Create 3D movement kernel with turbulence."""
        kernel = np.zeros((3, 3, 3))
        wind_magnitude = np.sqrt(wind_u**2 + wind_v**2 + wind_w**2)
        wind_scale = min(wind_magnitude / QuicFireConstants.U_REF, 1.0)
        turbulence = QuicFireConstants.TURBULENCE_FACTOR * wind_magnitude / QuicFireConstants.U_REF
        kernel[1, 1, 1] = QuicFireConstants.MOVEMENT_PROBS['stay'] * (1 - 0.5 * wind_scale)
        
        if wind_magnitude > 0:
            u_effect, v_effect, w_effect = wind_u / wind_magnitude, wind_v / wind_magnitude, wind_w / wind_magnitude
            for direction, (di, dj, dk) in [('east', (2, 1, 1)), ('west', (0, 1, 1)),
                                         ('north', (1, 2, 1)), ('south', (1, 0, 1)),
                                         ('up', (1, 1, 2)), ('down', (1, 1, 0))]:
                effect = {'east': u_effect, 'west': -u_effect, 'north': v_effect, 
                         'south': -v_effect, 'up': w_effect, 'down': -w_effect}[direction]
                kernel[di, dj, dk] = QuicFireConstants.MOVEMENT_PROBS[direction] * (1 + effect * wind_scale + turbulence)
        
        kernel = np.clip(kernel, 0.0, None)
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[1, 1, 1] = QuicFireConstants.MOVEMENT_PROBS['stay']
            kernel[0:3, 1, 1] = QuicFireConstants.MOVEMENT_PROBS['west']
            kernel[1, 0:3, 1] = QuicFireConstants.MOVEMENT_PROBS['south']
            kernel[1, 3, 1] = QuicFireConstants.MOVEMENT_PROBS['up']
            kernel[1, 1, 0:3] = QuicFireConstants.MOVEMENT_PROBS['down']

            kernel /= np.sum(kernel)
        return kernel

    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to EP movement."""
        self.EPs[0, :, :] *= QuicFireConstants.BOUNDARY_FACTORS['west']
        self.EPs[-1, :, :] *= QuicFireConstants.BOUNDARY_FACTORS['east']
        self.EPs[:, 0, :] *= QuicFireConstants.BOUNDARY_FACTORS['south']
        self.EPs[:, -1, :] *= QuicFireConstants.BOUNDARY_FACTORS['north']
        self.EPs[:, :, -1] *= QuicFireConstants.BOUNDARY_FACTORS['top']

    def _calculate_ignition_probability(self, mask: np.ndarray) -> np.ndarray:
        """Calculate ignition probability for given mask."""
        rho_f_local = self.fuel_density[mask]
        moisture = self.moisture_fraction[mask]
        wind_speed = np.sqrt(self.wind_manager.wind_u[mask]**2 + self.wind_manager.wind_v[mask]**2)
        
        Q_rad = QuicFireConstants.HEAT_FLUX * QuicFireConstants.RADIATIVE_FRACTION
        radiative_factor = (Q_rad * QuicFireConstants.H_SF) / (QuicFireConstants.C_P * QuicFireConstants.RHO_AIR * np.maximum(wind_speed, 0.1) * QuicFireConstants.T_AMBIENT)
        convective_factor = (QuicFireConstants.A_V * wind_speed**2) / np.maximum(
            QuicFireConstants.A_V_NORM * wind_speed**2, 1e-6)
        
        fuel_factor = np.where(rho_f_local > 0,
                             (rho_f_local / QuicFireConstants.C_IGSC) * (radiative_factor + convective_factor),
                             0.0)
        moisture_factor = np.maximum(0.1, 1.0 - 3.0 * moisture)
        P_base = (1 - np.exp(-fuel_factor)) * np.minimum(rho_f_local / QuicFireConstants.RHO_F0 * 2, 1.0) # DEBUG: Why is this multiplied by 2?
        return np.clip(P_base * moisture_factor, 0.0, 1.0)
    
    def _move_EPs(self) -> None:
        """Move EPs with vectorized operations and boundary conditions."""
        if np.sum(self.EPs) == 0:
            return
        
        padded_EPs = np.pad(self.EPs, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0.0)
        self.EPs = convolve(padded_EPs, self.kernel, mode='constant', cval=0.0)[1:-1, 1:-1, 1:-1]
        
        self._apply_boundary_conditions()

    def _process_EPs(self) -> None:
        """Process EPs for ignition, drying, and update moisture depletion fraction."""
        ep_mask = self.EPs > 0
        if not np.any(ep_mask):
            return
        ep_indices = np.where(ep_mask)
        ep_counts = np.minimum(self.EPs[ep_mask].astype(int), QuicFireConstants.MAX_EP_COUNT)
        P_ignition = self._calculate_ignition_probability(ep_mask)
        successful_ignitions = self.rng.binomial(ep_counts, P_ignition)
        
        # Calculate depletion probability, ensuring a minimum of 0.1
        depletion_prob = np.maximum(0.1, self.moisture_depletion_fraction[ep_mask])
        drying_mask = self.rng.random(len(ep_counts)) < depletion_prob
        
        # Compute drying rate
        drying_rate = QuicFireConstants.DRYING_RATE * successful_ignitions / np.maximum(ep_counts, 1)
        drying_rate *= QuicFireConstants.LAMBDA_EP / QuicFireConstants.LAMBDA_AIR
        drying_rate = np.where(drying_mask, drying_rate, 0.0)
        
        # Update moisture depletion fraction based on EP presence and heat flux
        heat_factor = self.heat_flux[ep_mask] / QuicFireConstants.HEAT_FLUX  # Normalize heat flux (0 to 1)
        ep_factor = np.minimum(ep_counts / QuicFireConstants.MAX_EP_COUNT, 1.0)  # Normalize EP count
        depletion_increase = QuicFireConstants.DRYING_RATE * 0.5 * (heat_factor + ep_factor)  # Gradual increase
        self.moisture_depletion_fraction[ep_mask] = np.clip(
            self.moisture_depletion_fraction[ep_mask] + depletion_increase,
            0.0,
            1.0
        )
        
        # Update moisture and fire states
        self._update_moisture_and_fires(ep_mask, drying_rate, successful_ignitions)
        self._handle_vertical_transport(ep_mask, ep_indices, successful_ignitions)
        self.EPs[ep_mask] *= (1 - QuicFireConstants.EP_DECAY_RATE) # Allow EPs to gradually decay

    def _update_moisture_and_fires(self, ep_mask: np.ndarray, drying_rate: np.ndarray, 
                                successful_ignitions: np.ndarray) -> None:
        """Update moisture and fire states with fuel threshold."""
        moisture_update = np.where(self.moisture_fraction[ep_mask] <= QuicFireConstants.MOISTURE_THRESHOLD,
                                self.moisture_fraction[ep_mask],
                                np.maximum(0, self.moisture_fraction[ep_mask] - drying_rate))
        self.moisture_fraction[ep_mask] = moisture_update
        new_fires = (moisture_update <= QuicFireConstants.MOISTURE_THRESHOLD) & \
                    (self.active_fire_time[ep_mask] <= 0) & \
                    (self.fuel_density[ep_mask] > QuicFireConstants.FUEL_DENSITY_THRESHOLD) & \
                    (successful_ignitions > 0)
        self.active_fire_time[ep_mask] = np.where(new_fires, QuicFireConstants.DT, self.active_fire_time[ep_mask])
        self.temperature[ep_mask] = np.where(new_fires, QuicFireConstants.T_FIRE, self.temperature[ep_mask])

    def _handle_vertical_transport(self, ep_mask: np.ndarray, ep_indices: Tuple[np.ndarray, ...], 
                                 successful_ignitions: np.ndarray) -> None:
        """Handle vertical transport of EPs."""
        valid_k = ep_indices[2] < self.nz - 1
        heat_mask = self.heat_flux[ep_mask] > 0.5 * QuicFireConstants.HEAT_FLUX
        vertical_mask = valid_k & heat_mask
        if np.any(vertical_mask):
            i_v, j_v, k_v = ep_indices[0][vertical_mask], ep_indices[1][vertical_mask], ep_indices[2][vertical_mask]
            self.moisture_fraction[i_v, j_v, k_v + 1] = np.maximum(
                0, self.moisture_fraction[i_v, j_v, k_v + 1] - QuicFireConstants.DRYING_RATE)
            self.EPs[i_v, j_v, k_v + 1] += 0.5 * successful_ignitions[vertical_mask]

    def _update_combustion(self) -> None:
        """
        Update combustion with oxygen limitation, using relative heights for calculations.
        """
        active_mask = self.active_fire_time > 0
        if not np.any(active_mask):
            return
        
        # Increment active_fire_time for active fires
        self.active_fire_time[active_mask] += QuicFireConstants.DT
        
        # Get the (i, j, k) indices for all currently active fire cells
        active_indices_i, active_indices_j, active_indices_k = np.where(active_mask)
        
        # Get the absolute z-coordinates for each active fire cell
        absolute_z_active_cells = self.z_levels[active_indices_k]
        
        # Get the terrain height for each active fire cell's (i, j) location
        terrain_height_at_active_cells = self.height_map[active_indices_i, active_indices_j]
        
        # Calculate the relative height for each active fire cell
        relative_heights_active_cells = absolute_z_active_cells - terrain_height_at_active_cells

        # Pass the relative heights to _calculate_oxygen_factor
        altitude_height_active_cells = relative_heights_active_cells + QuicFireConstants.BASE_ALTITUDE
        oxygen_factor = self._calculate_oxygen_factor_from_altitude(altitude_height_active_cells)
        
        # Calculate fire age and burnout fraction
        fire_age = self.active_fire_time[active_mask]
        burnout_fraction = fire_age / QuicFireConstants.T_BURNOUT
        
        # Calculate heat release, incorporating the oxygen factor
        heat_release = QuicFireConstants.HEAT_FLUX * (1 - burnout_fraction) * \
                       np.exp(-burnout_fraction / 2) * oxygen_factor
        
        heat_release_per_cell = heat_release * self.dx * self.dy  # kW/m² * m² = kW

        self._update_combustion_state(active_mask, heat_release_per_cell, burnout_fraction, 
                                      oxygen_factor, relative_heights_active_cells)

    def _calculate_oxygen_factor_from_altitude(
        self,
        absolute_altitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the oxygen availability factor purely based on absolute altitude
        using an exponential decay model, assuming an oxygen factor of 1.0 at 0 meters altitude.
        """
        # The oxygen factor is calculated as an exponential decay from 1.0 at 0 meters altitude.
        oxygen_factor = np.exp(-QuicFireConstants.OXYGEN_ALTITUDE_DECAY_RATE * absolute_altitudes)

        # Ensure oxygen factor does not go below zero
        oxygen_factor = np.maximum(0.0, oxygen_factor)
        
        return oxygen_factor

    def _update_combustion_state(self, active_mask: np.ndarray, heat_release_per_cell: np.ndarray, 
                                burnout_fraction: np.ndarray, oxygen_factor: np.ndarray, 
                                heights: np.ndarray) -> None:
        """Update combustion-related state arrays and zero depleted fuel."""
        new_eps = QuicFireConstants.EMISSION_RATE * heat_release_per_cell * QuicFireConstants.DT
        self.EPs[active_mask] += new_eps
        
        # Get indices of active cells
        active_indices = np.where(active_mask)
        i, j, k = active_indices
        dz_active = self.dz[k]
        
        # Correct fuel consumption (kg/m²/s * s / m = kg/m³)
        fuel_consumed_volumetric_dt = np.minimum(
            (QuicFireConstants.CONSUMPTION_RATE * QuicFireConstants.DT * 
            (1 - burnout_fraction) * oxygen_factor) / dz_active,
            self.fuel_density[active_mask] * 0.1  # Consume up to 10% per timestep
        )
        self.fuel_density[active_mask] -= fuel_consumed_volumetric_dt
        self.fuel_density[active_mask] = np.where(self.fuel_density[active_mask] < QuicFireConstants.FUEL_DENSITY_THRESHOLD, 0, self.fuel_density[active_mask])
        
        ### --------------------------- DEBUGGING AND CALIBRATION ------------------------------- ###
        if self.save_interval:
            # Calculate the actual local mass burn rate (kg/m^2/s) for each active cell.
            # This requires knowing the dz for each active cell.
            active_indices_k = np.where(active_mask)[2] # Get k-indices of active cells
            dz_active_cells = self.dz[active_indices_k] # Get dz for each active cell

            # local_mass_burn_rates is (kg/m^3) * m / s = kg/m^2/s
            local_mass_burn_rates = (fuel_consumed_volumetric_dt * dz_active_cells) / QuicFireConstants.DT

            # Calculate the average over active_mask cells.
            avg_consumption_rate_per_area = np.mean(local_mass_burn_rates) if np.any(local_mass_burn_rates) else 0.0

            # Store for calibration
            self.consumption_rate_history.append(avg_consumption_rate_per_area)

            # DEBUG: Use relative heights to terrain
            # This is only for metrics
            strata = np.where(heights <= QuicFireConstants.HEIGHT_THRESHOLDS['surface'], 'surface',
                    np.where(heights <= QuicFireConstants.HEIGHT_THRESHOLDS['midstory'], 'midstory', 'canopy'))
            for strata_name in ['surface', 'midstory', 'canopy']:
                strata_dz = dz_active_cells[strata == strata_name]
                self.consumption[strata_name] += np.sum(fuel_consumed_volumetric_dt[strata == strata_name] * self.dx * self.dy * strata_dz)
        
        ### ------------------------------------------------------------------------------------- ###
        
        # Physics
        self.temperature[active_mask] = QuicFireConstants.T_AMBIENT + (QuicFireConstants.T_FIRE - QuicFireConstants.T_AMBIENT) * (1 - burnout_fraction)
        self.heat_flux[active_mask] = heat_release_per_cell
        extinguished = (self.active_fire_time[active_mask] >= QuicFireConstants.T_BURNOUT) | (self.fuel_density[active_mask] <= 0)
        self.active_fire_time[active_mask] = np.where(extinguished, 0, self.active_fire_time[active_mask])
        self.temperature[active_mask] = np.where(extinguished, QuicFireConstants.T_AMBIENT, self.temperature[active_mask])
        self.heat_flux[active_mask] = np.where(extinguished, 0, self.heat_flux[active_mask])

    def _transport_EPs_vertically(self, w_1_full_3d: np.ndarray, active_fires: np.ndarray) -> None:
        """Transport EPs vertically based on plume dynamics."""
        # Check if all vertical velocities are non-positive across *all* cells where EPs might exist.
        if np.all(w_1_full_3d <= 0):
            return

        # Find the indices (i, j, k) of cells that are active and have EPs
        active_indices_tuple = np.where(active_fires & (self.EPs > 0))
        i_all_active, j_all_active, k_all_active = active_indices_tuple

        # Get the w_1 values specifically for these active cells
        w_1_active_cells = w_1_full_3d[i_all_active, j_all_active, k_all_active]

        # Filter for cells where vertical transport UP is possible (not at the top layer)
        valid_mask_for_upward_transport = k_all_active < self.nz - 1

        # Apply this mask to all collected indices and their corresponding w_1 values
        i_transport, j_transport, k_transport = \
            i_all_active[valid_mask_for_upward_transport], \
            j_all_active[valid_mask_for_upward_transport], \
            k_all_active[valid_mask_for_upward_transport]

        w_1_transport = w_1_active_cells[valid_mask_for_upward_transport]

        if len(i_transport) == 0:
            # All EPs are already in the top layer
            return
        
        # Calculate EPs to move
        EPs_to_move = self.EPs[i_transport, j_transport, k_transport] * w_1_transport * QuicFireConstants.DT
        # Note: DT_substep is often needed for transport calculations, not the overall QuicFireConstants.DT

        # Ensure EPs_to_move doesn't exceed available EPs in source cell
        EPs_to_move = np.minimum(EPs_to_move, self.EPs[i_transport, j_transport, k_transport])

        # Apply the transfer
        self.EPs[i_transport, j_transport, k_transport] -= EPs_to_move
        self.EPs[i_transport, j_transport, k_transport + 1] += EPs_to_move

    def _compute_plume_dynamics(self) -> None:
        """
        Computes plume dynamics parameters necessary for transporting embers and
        calls the transport function. This function does not return any data.
        """
        active_fires = self.active_fire_time > 0
        
        # Initialize a 3D array for w_1, filled with zeros
        w_1_full_grid = np.zeros_like(self.temperature)

        if not np.any(active_fires):
            # Call the transport function with a zero grid if no active fires
            self._transport_EPs_vertically(w_1_full_grid, active_fires)
            return

        # Get 3D indices of active cells
        i, j, k = np.where(active_fires) 

        # Calculate properties for active cells only
        relative_heights_active = self.z_levels[k] - self.height_map[i, j]
        heat_release_per_cell = self.heat_flux[active_fires] * self.dx * self.dy  # kW
        total_heat = np.sum(heat_release_per_cell) # kW

        if total_heat > 0:
            Q_cell = heat_release_per_cell * 1000  # W
            F_R_active = (QuicFireConstants.GRAVITY * Q_cell) / (QuicFireConstants.RHO_AIR * QuicFireConstants.C_P * QuicFireConstants.T_AMBIENT)  # m^4/s^3

            w_1_values_active = np.sqrt((2/3) * F_R_active / np.maximum(0.1, relative_heights_active))
            
            # Place these calculated w_1 values back into the full 3D grid at the active cell locations
            w_1_full_grid[i, j, k] = w_1_values_active
            
        # Pass the full 3D w_1_full_grid to the transport function
        self._transport_EPs_vertically(w_1_full_grid, active_fires)
    
    def _get_terrain_z_index(self, sim_x: int, sim_y: int) -> int:
        """
        Returns the simulation's Z-index corresponding to the terrain height
        at the given simulation (x, y) coordinates.
        """
        if self.height_map is None:
            raise RuntimeError("height_map has not been initialized. "
                               "Ensure _generate_3d_fuel_grid or similar setup is called.")
        
        if not (0 <= sim_x < self.nx and 0 <= sim_y < self.ny):
            raise IndexError(f"Simulation coordinates ({sim_x}, {sim_y}) are out of bounds "
                             f"for grid size ({self.nx}, {self.ny}).")

        current_terrain_height = self.height_map[sim_x, sim_y]
        
        # Find the lowest z-grid index (k_ground) that is at or above the terrain height.
        # 'side=right' means values equal to current_terrain_height go to the next bin, so it's above.
        k_ground = np.searchsorted(self.z_levels, current_terrain_height, side='right')
        
        # Ensure k_ground is within bounds [0, self.nz - 1]
        k_ground = min(k_ground, self.nz - 1)
        k_ground = max(k_ground, 0) # Ensure it's not negative if terrain is below lowest z-layer

        return int(k_ground)
    
    def _get_fire_positions(self) -> np.ndarray:
        """Get active fire positions and properties."""
        fire_mask = self.active_fire_time > 0
        if not np.any(fire_mask):
            return np.array([], dtype=[('position', float, 3), ('intensity', float), 
                                    ('temperature', float), ('fuel_remaining', float), 
                                    ('fire_age', float)])
        i, j, k = np.where(fire_mask)
        return np.array([
            ((i[idx] * self.dx, j[idx] * self.dy, self.z_levels[k[idx]]), 
            self.heat_flux[i[idx], j[idx], k[idx]],
            self.temperature[i[idx], j[idx], k[idx]],
            self.fuel_density[i[idx], j[idx], k[idx]],
            self.active_fire_time[i[idx], j[idx], k[idx]])
            for idx in range(len(i))
        ], dtype=[('position', float, 3), ('intensity', float), ('temperature', float), 
                ('fuel_remaining', float), ('fire_age', float)])

    def _get_ep_positions(self) -> List[Dict]:
        """Get ember position data."""
        ep_positions = []
        if np.sum(self.EPs) > 0:
            ep_indices = np.where(self.EPs > 0.1)
            for idx in range(min(1000, len(ep_indices[0]))):
                i, j, k = ep_indices[0][idx], ep_indices[1][idx], ep_indices[2][idx]
                ep_positions.append({
                    'position': (i * self.dx, j * self.dy, self.z_levels[k]),
                    'density': self.EPs[i, j, k]
                })
        return ep_positions
    
    def get_diagnostics(self) -> Dict:
        """Get simulation diagnostics."""
        return {
            'active_fires': np.sum(self.active_fire_time > 0),
            'total_eps': np.sum(self.EPs),
            'total_fuel_remaining': np.sum(self.fuel_density),
            'average_moisture': np.mean(self.moisture_fraction),
            'max_temperature': np.max(self.temperature),
            'total_heat_release': np.sum(self.heat_flux),
            'consumption': self.consumption.copy() # Not updated unless save_interval is active
        }
    
    def _get_xr_plume_dynamics(self) -> list[dict]:
        """
        Calculate plume dynamics for wildfire clusters using Briggs plume rise and McCaffrey velocity profiles.
        Returns list of plume dictionaries compatible with Unity shader (PlumeCellData).
        """

        active_fires_mask = self.active_fire_time > 0
        if not np.any(active_fires_mask):
            return []

        labeled_array, num_features = label(active_fires_mask)
        plume_data_list = []

        for feature_id in range(1, num_features + 1):
            cluster_mask = (labeled_array == feature_id)
            i_c, j_c, k_c = np.where(cluster_mask)

            if len(i_c) == 0:
                continue

            # --- Plume Origin ---
            # Centroid of fire cluster in grid coordinates, converted to world space
            mean_i = np.mean(i_c)
            mean_j = np.mean(j_c)
            mean_k_level = np.mean(self.height_map[i_c, j_c])

            plume_origin_x = mean_i * self.dx
            plume_origin_y = mean_j * self.dy
            plume_origin_z = mean_k_level

            # --- Heat Release ---
            # Total heat (kW) from cluster, integrating heat flux (kW/m²) over area
            heat_release_cluster_cells = self.heat_flux[cluster_mask] * self.dx * self.dy
            total_heat_cluster = np.sum(heat_release_cluster_cells)
            if total_heat_cluster <= 0:
                continue

            # --- Fire Geometry ---
            # Equivalent diameter for plume scaling (McCaffrey 1979, assumes circular source)
            fire_area = len(i_c) * self.dx * self.dy  # m²
            fire_diameter = 2 * np.sqrt(fire_area / np.pi)  # m

            # --- Buoyancy Flux ---
            # F = g * Q / (ρ Cp T_amb) in m⁴/s³, where Q is in W (Briggs 1975)
            Q_total = total_heat_cluster * 1000.0  # Convert kW to W
            F = QuicFireConstants.GRAVITY * Q_total / (QuicFireConstants.RHO_AIR * QuicFireConstants.C_P * QuicFireConstants.T_AMBIENT)

            # --- Wind Speed ---
            # Mean horizontal wind speed over cluster (for Briggs rise and tilt)
            wind_speed_cluster = np.sqrt(np.mean(self.wind_manager.wind_u[cluster_mask])**2 +
                                        np.mean(self.wind_manager.wind_v[cluster_mask])**2)

            # --- Plume Height (h_sub) ---
            # Briggs neutral atmosphere, bent-over plume: h = coeff * F^{1/3} x^{2/3} / u
            # Evaluate at x = fire_diameter (near-field rise, typical for visualization)
            # Ref: Briggs 1975, "Plume Rise Predictions"
            length_floor = QuicFireConstants.PLUME_LENGTH_FLOOR
            x_ref = max(fire_diameter, length_floor)  # Avoid div0
            if wind_speed_cluster > QuicFireConstants.BRIGGS_WIND_THRESHOLD:
                h_sub_plume = QuicFireConstants.BRIGGS_RISE_COEFF * pow(F, 1.0/3.0) * pow(x_ref, 2.0/3.0) / wind_speed_cluster
            else:
                # Low wind: Use buoyant rise ~ F^{2/5} (Morton 1956, simplified)
                h_sub_plume = QuicFireConstants.MORTON_RISE_COEFF * pow(F, 2.0/5.0)
            h_sub_plume = max(h_sub_plume, fire_diameter * QuicFireConstants.PLUME_MIN_RISE_DIAMETERS)
            h_sub_plume = min(h_sub_plume, QuicFireConstants.PLUME_MAX_HEIGHT)

            # --- Plume Length (ell) ---
            # Horizontal projection at h_sub: x where h(x) = h_sub
            # From Briggs: x ≈ (h * u / (coeff * F^{1/3}))^{3/2}
            if wind_speed_cluster > QuicFireConstants.BRIGGS_WIND_THRESHOLD:
                ell_plume = pow(h_sub_plume * wind_speed_cluster / (QuicFireConstants.BRIGGS_RISE_COEFF * pow(F, 1.0/3.0)), 1.5)
                ell_plume = max(ell_plume, h_sub_plume)  # Ensure at least vertical rise
            else:
                ell_plume = h_sub_plume * QuicFireConstants.PLUME_CALM_TILT_FACTOR
            ell_plume = min(ell_plume, h_sub_plume * QuicFireConstants.PLUME_MAX_TILT_FACTOR)

            # --- Vertical Velocity (plume_w) ---
            # Centerline velocity at z = fire_diameter: w ∝ (F / z)^{1/3} (McCaffrey 1979)
            z_ref = max(fire_diameter, length_floor)
            w_cl = QuicFireConstants.MCCAFFREY_VELOCITY_COEFF * pow(F / z_ref, 1.0/3.0)
            w_cl = min(w_cl, QuicFireConstants.PLUME_MAX_VERTICAL_VELOCITY)

            # Wind components (mean over cluster, no double-counting vertical)
            wind_u_mean_cluster = np.mean(self.wind_manager.wind_u[cluster_mask])
            wind_v_mean_cluster = np.mean(self.wind_manager.wind_v[cluster_mask])

            # --- Turbulence Factor (k) ---
            # Scale with Froude number: Fr = w_cl² / (g * fire_diameter)
            # Base plus buoyancy and wind shear, then capped (Morton 1956)
            fr = w_cl * w_cl / (QuicFireConstants.GRAVITY * max(fire_diameter, length_floor))
            k_val_plume = QuicFireConstants.PLUME_K_BASE + QuicFireConstants.PLUME_K_FROUDE_WEIGHT * min(fr, QuicFireConstants.PLUME_K_FROUDE_CAP)
            shear = wind_speed_cluster / QuicFireConstants.PLUME_K_SHEAR_WIND
            k_val_plume += QuicFireConstants.PLUME_K_SHEAR_WEIGHT * min(shear, 1.0)
            k_val_plume = min(k_val_plume, QuicFireConstants.PLUME_K_MAX)

            # --- Output ---
            plume_data_list.append({
                'origin_position': [plume_origin_x, plume_origin_y, plume_origin_z],
                'plume_u': wind_u_mean_cluster,
                'plume_v': wind_v_mean_cluster,
                'plume_w': w_cl,
                'total_heat': total_heat_cluster,
                'h_sub': h_sub_plume,
                'ell': ell_plume,
                'k': k_val_plume
            })

        return plume_data_list

    def get_xr_visualization_data(self) -> Dict[str, Any]:
        """
        Prepares raw visualization data from the engine for Unity.
        This data is formatted for easy serialization to JSON.
        """
        # 1. Get and process fire positions
        # The helper function returns a structured numpy array. We need to convert it.
        fire_structured_array = self._get_fire_positions()
        fire_positions_list = []
        if fire_structured_array.size > 0:
            # Iterate over the structured array and convert each entry to a dictionary
            for fire in fire_structured_array:
                fire_positions_list.append({
                    'position': fire['position'].tolist(),
                    'intensity': fire['intensity'].item(),
                    'temperature': fire['temperature'].item(),
                    'fuel_remaining': fire['fuel_remaining'].item(),
                    'fire_age': fire['fire_age'].item(),
                })

        # 2. Get and process ember positions (already in a serializable format)
        ep_positions = self._get_ep_positions()
        
        # 3. Get and process wind field data
        # The helper function returns a dict of numpy arrays. We must convert these.
        # wind_data_dict = self.wind_manager.get_wind_field()
        
        # wind_field_for_json = {}
        # if wind_data_dict is not None:
        #     # Get shape from one of the arrays
        #     wind_shape = wind_data_dict['u'].shape
            
        #     wind_field_for_json = {
        #         'u': wind_data_dict['u'].flatten().tolist(),
        #         'v': wind_data_dict['v'].flatten().tolist(),
        #         'w': wind_data_dict['w'].flatten().tolist(),
        #         'shape': wind_shape  # Provide shape so Unity can reconstruct the 3D grid
        #     }

        # 4. Get plume and diagnostics (assuming they are already JSON-friendly)
        plume_data = self._get_xr_plume_dynamics()

        # 5. Assemble and return the final, JSON-ready dictionary
        return {
            'fires': fire_positions_list,
            'embers': ep_positions,
            'plumes': plume_data,
            # 'wind_field': wind_field_for_json,
        }
    
    def apply_firefighting_action(
        self,
        action_type: Literal['water', 'retardant', 'firebreak', 'air_support'],
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        z_min: int = 0,
        z_max: Optional[int] = None,
        intensity: float = 1.0,
    ) -> None:
        """
        Applies a firefighting action to a specified region of the simulation grid.

        Args:
            action_type (Literal): The type of action ('water', 'retardant', 'firebreak', 'air_support').
            x_min, y_min (int): Minimum x and y coordinates (inclusive) of the target region.
                                These should be in the 'grid_size' scale from the AI, 
                                and will be scaled to 'nx', 'ny' internally.
            x_max, y_max (int): Maximum x and y coordinates (inclusive) of the target region.
            z_min (int): Minimum z coordinate (inclusive) of the target region. Defaults to 0 (surface).
            z_max (Optional[int]): Maximum z coordinate (inclusive) of the target region. 
                                   Defaults to the top of the grid if None.
            intensity (float): Strength of the application (e.g., for water/retardant/air_support).
                                1.0 means full effect, lower means less.
        """

        # Ensure coordinates are within grid bounds (as before)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(self.nx - 1, x_max)
        y_max = min(self.ny - 1, y_max)
        z_max = min(self.nz - 1, z_max)
        
        target_slice = (
            slice(x_min, x_max + 1),
            slice(y_min, y_max + 1),
            slice(z_min, z_max + 1)
        )

        if action_type in ['water', 'retardant', 'air_support']:
            # 1. Increase moisture fraction (prevents re-ignition, reduces flammability)
            moisture_increase_amount = 0.0
            if action_type == 'water':
                moisture_increase_amount = QuicFireConstants.MOISTURE_INCREASE_WATER * intensity
            elif action_type == 'retardant':
                moisture_increase_amount = QuicFireConstants.MOISTURE_INCREASE_RETARDANT * intensity
            elif action_type == 'air_support':
                moisture_increase_amount = QuicFireConstants.MOISTURE_INCREASE_AIR_SUPPORT * intensity
            
            self.moisture_fraction[target_slice] = np.clip(
                self.moisture_fraction[target_slice] + moisture_increase_amount, 0.0, 1.0
            )

            # 2. Immediately extinguish the fire (active suppression)
            # By setting heat_flux to zero, the fire stops generating heat.
            self.heat_flux[target_slice] = 0.0
            
            # Reset active_fire_time to zero. This is crucial to prevent "ghost" fire age
            # and ensures that if fire reignites, it starts fresh.
            self.active_fire_time[target_slice] = 0.0
            
            # Reset temperature to ambient where fire is extinguished for consistency.
            self.temperature[target_slice] = QuicFireConstants.T_AMBIENT

        elif action_type == 'firebreak':
            # Set fuel density to zero in the specified region
            self.fuel_density[target_slice] = 0.0
            
            # If fuel is removed, no fire can exist or generate heat.
            self.active_fire_time[target_slice] = 0.0 # Definitely reset fire age if fuel removed
            self.heat_flux[target_slice] = 0.0 # No heat from no fuel
            self.temperature[target_slice] = QuicFireConstants.T_AMBIENT

        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        # Ensure all affected fields remain within valid physical bounds
        self.moisture_fraction = np.clip(self.moisture_fraction, 0.0, 1.0)
        self.fuel_density = np.clip(self.fuel_density, 0.0, QuicFireConstants.RHO_F0)
        self.heat_flux = np.clip(self.heat_flux, 0.0, QuicFireConstants.HEAT_FLUX * self.nx * self.ny) 
        # temperature might not need clipping if always set to T_AMBIENT or left to engine

    def step(self) -> None:
        """
        Advances the simulation by one QuicFireConstants.DT time step.
        This function encapsulates all the physics updates for a single time step.
        """
        # 1. Update wind field (if any scheduled updates)
        for update in self.wind_updates:

            # DEBUG: Is it worth creating a kernel for each z-level to have more accurate wind?
            if update.get('time', float('inf')) <= self.current_simulation_time:
                # Update wind field
                self.wind_manager.update(update.get('speed'), update.get('direction'), self.current_simulation_time, self.active_fire_time, self.heat_flux)
                
                # Average wind and update movement kernel
                wind_avg = self.wind_manager.calculate_average_wind(self.active_fire_time)
                self.kernel = self._create_movement_kernel(*wind_avg)

                # Remove applied update from schedule
                self.wind_updates.remove(update)

        # 2. Update the movement kernel based on current average wind
        wind_avg = self.wind_manager.calculate_average_wind(self.active_fire_time)
        self.kernel = self._create_movement_kernel(*wind_avg)

        # 3. Process EPs (move, ignite, dry fuel)
        self._move_EPs()
        self._process_EPs()

        # 4. Update combustion and associated processes (fuel consumption, temp, heat flux)
        self._update_combustion()

        # 5. Compute plume dynamics and vertical EP transport
        self._compute_plume_dynamics()

        # 6. Advance simulation time
        self.current_simulation_time += QuicFireConstants.DT

    def run_simulation(self, 
                    save_interval: Optional[int] = None,
                    verbose=False,
                    ) -> Dict:
        """Run the QUIC-Fire simulation with fuel monitoring and dynamic wind updates."""

        self.save_interval = save_interval
        start_time = time.time()
        saved_data = []

        diagnostics = self.get_diagnostics()

        starting_fuel = diagnostics['total_fuel_remaining']

        if verbose:
            print(f"Starting QUIC-Fire simulation... Grid size: {self.nx} x {self.ny} x {self.nz}")
            print(f"Initial state: {diagnostics}")
        
        try:
            for t in range(0, self.total_time, int(QuicFireConstants.DT)):
                step_start = time.time()
                
                # Step Simulation
                self.step()
                step_time = time.time() - step_start
                
                # Update diagnostics after step
                diagnostics = self.get_diagnostics()

                # Return diagnostics and visualization data at intervals
                if self.current_simulation_time % QuicFireConstants.PROGRESS_INTERVAL == 0 and verbose:
                    elapsed = time.time() - start_time
                    print(f"t={t:4d}s: Fires={diagnostics['active_fires']:4d}, "
                        f"EPs={diagnostics['total_eps']:8.1f}, "
                        f"Heat={diagnostics['total_heat_release']:6.1f}kW, "
                        f"Fuel={diagnostics['total_fuel_remaining']:.1f}kg, "
                        f"Step={step_time:.3f}s, Elapsed={elapsed:.1f}s")
                
                if self.save_interval and t % self.save_interval == 0:
                    saved_data.append({'time': t, 'data': self.get_xr_visualization_data()})
                
                if self.current_simulation_time > 300 and (diagnostics['active_fires'] == 0 or diagnostics['total_fuel_remaining'] < 0.01 * self.initial_fuel):
                    print(f"✓ {'Fire extinguished' if diagnostics['active_fires'] == 0 else 'Fuel exhausted'} at t={self.current_simulation_time}s")
                    break
        
        except KeyboardInterrupt:
            print("⚠ Simulation interrupted by user")
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
        
        # Final diagnostics and cleanup
        final_time = time.time() - start_time
        final_diagnostics = self.get_diagnostics()

        ending_fuel = final_diagnostics['total_fuel_remaining']
        burn_rate = (starting_fuel - ending_fuel) / self.current_simulation_time

        if verbose:
            print(f"\n✓ Simulation completed in {final_time:.1f}s")
            print(f"Final state: {final_diagnostics}")
            print(f"Burn Rate: {burn_rate} kg/s")
        
        result = self.get_xr_visualization_data()
        if saved_data:
            result['time_series'] = saved_data
            result['consumption_rate_history'] = self.consumption_rate_history

        return result
    
    def close(self):
        """
        Cleans up resources held by the QuicFireEngine.
        This method is designed to be idempotent (can be called multiple times safely).
        """
        if not hasattr(self, '_is_closed') or not self._is_closed:
            print("Closing QuicFireEngine instance...")

            # 1. Explicitly release references to large NumPy arrays
            # This aids garbage collection, especially for large simulations
            self.height_map = None
            self.fuel_density = None
            self.fuel_type_grid = None # Assuming this is also a large grid
            self.fire_risk = None      # Assuming this is also a large grid
            self.active_fire_time = None # This is likely updated during simulation

            # 2. Call close/destroy on any sub-managers if they have such methods
            if hasattr(self.wind_manager, 'close') and callable(self.wind_manager.close):
                self.wind_manager.close()
            if hasattr(self.ignition_manager, 'close') and callable(self.ignition_manager.close):
                self.ignition_manager.close()

            # Set manager references to None
            self.map_generator = None
            self.wind_manager = None
            self.ignition_manager = None
            
            # Clear config and other attributes if desired, though less critical
            self.config = None
            self.rng = None

            # Set a flag to indicate that the engine has been closed
            self._is_closed = True
            print("QuicFireEngine closed successfully.")
        else:
            print("QuicFireEngine is already closed.")