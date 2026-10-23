import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from scipy.ndimage import uniform_filter
from QuicFire.QuicFireConstants import QuicFireConstants

class QuicFireWindFieldManager:
    """Manages the 3D wind field for the QUIC-Fire simulation."""
    
    def __init__(self, rng: np.random.Generator, nx: int, ny: int, nz: int, z_levels: np.ndarray, 
                 dx: int, dy: int, fuel_density: np.ndarray, height_map: np.ndarray) -> None:
        """Initialize wind field manager.
        
        Args:
            rng: Random number generator for stochastic processes.
            nx: Number of grid points in x-direction.
            ny: Number of grid points in y-direction.
            nz: Number of grid points in z-direction.
            z_levels: Array of vertical grid heights.
            dx: Grid spacing in x-direction (m).
            dy: Grid spacing in y-direction (m).
            fuel_density: 3D array of fuel density values.
            height_map: 2D array of terrain heights.
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.z_levels = z_levels
        self.dx = dx
        self.dy = dy
        self.fuel_density = fuel_density
        self.height_map = height_map
        self.rng = rng
        
        # Compute terrain derivatives once
        self.grad_x = np.gradient(self.height_map, self.dx, axis=0)
        self.grad_y = np.gradient(self.height_map, self.dy, axis=1)
        self.slope_magnitude = np.sqrt(self.grad_x**2 + self.grad_y**2)
        
        # Validate inputs
        self._validate_inputs()
        
        self.u_star = self._calculate_u_star()
        self._global_wind_avg = None

        # Initialize wind field arrays
        self.wind_u = np.zeros((self.nx, self.ny, self.nz))
        self.wind_v = np.zeros((self.nx, self.ny, self.nz))
        self.wind_w = np.zeros((self.nx, self.ny, self.nz))

        # Initial call to populate the wind field
        self.initialize()

        # Internal flag
        self._is_closed = False
    
    def _validate_inputs(self) -> None:
        """Validate input parameters and arrays."""
        assert self.nx > 0 and self.ny > 0 and self.nz > 0, "Grid dimensions must be positive"
        assert self.dx > 0 and self.dy > 0, "Grid spacing must be positive"
        
        assert self.fuel_density.shape == (self.nx, self.ny, self.nz), f"Fuel density shape {self.fuel_density.shape} doesn't match grid ({self.nx}, {self.ny}, {self.nz})"
        assert self.height_map.shape == (self.nx, self.ny), f"Terrain height map shape {self.height_map.shape} doesn't match grid ({self.nx}, {self.ny})"
        assert self.grad_x.shape == (self.nx, self.ny), f"grad_x shape {self.grad_x.shape} doesn't match grid ({self.nx}, {self.ny})"
        assert self.grad_y.shape == (self.nx, self.ny), f"grad_y shape {self.grad_y.shape} doesn't match grid ({self.nx}, {self.ny})"
        assert self.slope_magnitude.shape == (self.nx, self.ny), f"slope_magnitude shape {self.slope_magnitude.shape} doesn't match grid ({self.nx}, {self.ny})"
        assert len(self.z_levels) == self.nz, f"Z array length {len(self.z_levels)} doesn't match nz {self.nz}"
        
        assert np.all(self.fuel_density >= 0), "Fuel density must be non-negative"
        assert np.all(self.height_map >= 0), "Terrain heights must be non-negative"
        assert np.all(self.z_levels >= 0), "Z coordinates must be non-negative"
        assert np.any(self.z_levels > QuicFireConstants.Z0), "At least one z_level must be greater than Z0"
        assert np.all(np.diff(self.z_levels) > 0), "Z coordinates must be monotonically increasing"
        assert np.all(self.slope_magnitude >= 0), "Slope magnitude must be non-negative"
        
        assert 0.001 <= QuicFireConstants.Z0 <= 10.0, "Roughness length Z0 out of valid range (0.001-10.0 m)"
        assert 0.3 <= QuicFireConstants.KAPPA <= 0.5, "Von Karman constant KAPPA out of valid range (0.3-0.5)"
        assert 0.1 <= QuicFireConstants.U_REF <= 50.0, "Reference wind speed out of valid range (0.1-50.0 m/s)"
        assert 1.0 <= QuicFireConstants.Z_REF <= 100.0, "Reference height out of valid range (1.0-100.0 m)"
        assert QuicFireConstants.Z_REF > QuicFireConstants.Z0, "Z_REF must be greater than Z0"
        assert 0.0 <= QuicFireConstants.TURBULENCE_FACTOR <= 1.0, "Turbulence factor out of valid range (0.0-1.0)"
        assert 0.0 <= QuicFireConstants.WIND_UPDRAFT_FACTOR <= 10.0, "Wind updraft factor out of valid range (0.0-10.0)"
        assert QuicFireConstants.HEAT_FLUX > 0, "Heat flux must be positive for updraft calculation"
        assert 'canopy' in QuicFireConstants.FUEL_HEIGHTS, "FUEL_HEIGHTS must be a dict with 'canopy' key"
        assert QuicFireConstants.WIND_FIELD_SIZE_LIMIT > 0, "WIND_FIELD_SIZE_LIMIT must be positive"

        print("✓ WindFieldManager: Input validation passed")
        
    def _calculate_u_star(self) -> float:
        """Calculate friction velocity."""
        denom = np.log(QuicFireConstants.Z_REF / QuicFireConstants.Z0)
        if np.isclose(denom, 0, atol=1e-6):
            print("Warning: Z_REF ≈ Z0, using small u_star value")
            return 0.01
        return QuicFireConstants.U_REF * QuicFireConstants.KAPPA / denom
    
    def initialize(self, wind_speed: float = None, wind_direction: float = 0.0, t: float = 0.0, 
                   active_fire_time: Optional[np.ndarray] = None, heat_flux: Optional[np.ndarray] = None) -> None:
        """Initialize 3D wind field with user-specified speed, direction, and dynamic behavior."""
        self._global_wind_avg = None
        
        wind_speed_ref = wind_speed or QuicFireConstants.U_REF
        
        # Convert direction to radians (0° = north, 90° = east)
        theta = np.radians(wind_direction)
        
        # Dynamic gust variation
        gust_period = 300.0
        gust_amplitude = 0.1 * wind_speed_ref
        direction_variation = np.radians(15.0) * np.sin(2 * np.pi * t / gust_period)
        speed_variation = gust_amplitude * np.sin(2 * np.pi * t / gust_period)
        effective_speed = wind_speed_ref + speed_variation
        effective_theta = theta + direction_variation
        
        # Vectorized wind profile factors, only for z_levels > 0
        denom = np.log(QuicFireConstants.Z_REF / QuicFireConstants.Z0)
        profile_factors_z = np.zeros_like(self.z_levels)
        if np.isclose(denom, 0, atol=1e-6):
            profile_factors_z[:] = 1.0
        else:
            valid_z = self.z_levels > 0
            profile_factors_z[valid_z] = np.log(self.z_levels[valid_z] / QuicFireConstants.Z0) / denom
        
        # Base wind components
        base_wind_speed = effective_speed * profile_factors_z  # shape (nz,)
        base_wind_u = base_wind_speed * np.sin(effective_theta)  # shape (nz,)
        base_wind_v = base_wind_speed * np.cos(effective_theta)  # shape (nz,)
        
        # Slope factors (using precomputed grad_x, grad_y, slope_magnitude)
        slope_angle = np.arctan(self.slope_magnitude)
        with np.errstate(divide='ignore', invalid='ignore'):
            slope_u_factor = np.where(self.slope_magnitude > 0, 1.0 + 0.3 * (self.grad_x / self.slope_magnitude) * np.sin(slope_angle), 1.0)
            slope_v_factor = np.where(self.slope_magnitude > 0, 1.0 + 0.3 * (self.grad_y / self.slope_magnitude) * np.sin(slope_angle), 1.0)
        slope_w_factor = 1.0 + 2.0 * self.slope_magnitude
        
        # Channeling effects
        sum_neighbors = uniform_filter(self.height_map, size=3, mode='constant', cval=0.0) * 9 - self.height_map
        avg_neighbor_height = sum_neighbors / 8
        height_diff = self.height_map - avg_neighbor_height
        channeling_factor = np.where(height_diff < -5.0, 1.2, np.where(height_diff > 5.0, 0.8, 1.0))
        
        # Local roughness
        max_surface_levels = min(5, self.nz)
        surface_fuel = np.sum(self.fuel_density[:, :, :max_surface_levels], axis=2)
        local_roughness = np.minimum(0.01 + 0.1 * (surface_fuel / 10.0), 2.0)
        
        # Height above ground level
        height_agl = self.z_levels[None, None, :] - self.height_map[:, :, None]
        
        # Roughness factor
        z0 = np.maximum(local_roughness[:, :, None] * 0.1, 0.001)
        effective_height = np.maximum(height_agl, z0)
        with np.errstate(divide='ignore', invalid='ignore'):
            profile_factor = np.log(effective_height / z0) / np.log(10.0 / z0)
        roughness_factor = np.where(height_agl < 10.0, np.clip(profile_factor, 0.1, 1.0), 1.0)
        
        # Orographic effects
        wind_speed_k = np.sqrt(base_wind_u**2 + base_wind_v**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            upwind_grad = (self.grad_x[:, :, None] * base_wind_u[None, None, :] + 
                           self.grad_y[:, :, None] * base_wind_v[None, None, :]) / wind_speed_k[None, None, :]
            upwind_grad = np.where(wind_speed_k[None, None, :] > 0.1, upwind_grad, 0.0)
        lift_factor = 1.0 + 3.0 * upwind_grad * np.exp(-np.maximum(0.0, height_agl) / 100.0)
        orographic_factor = np.clip(lift_factor, 0.1, 3.0)
        
        # Modified wind components
        modified_u = base_wind_u[None, None, :] * roughness_factor * slope_u_factor[:, :, None] * channeling_factor[:, :, None]
        modified_v = base_wind_v[None, None, :] * roughness_factor * slope_v_factor[:, :, None] * channeling_factor[:, :, None]
        
        # Turbulence
        turbulence_scale = QuicFireConstants.TURBULENCE_FACTOR * np.sqrt(modified_u**2 + modified_v**2)
        turbulence_u = self.rng.normal(0, turbulence_scale)
        turbulence_v = self.rng.normal(0, turbulence_scale)
        
        self.wind_u = modified_u + turbulence_u
        self.wind_v = modified_v + turbulence_v
        self.wind_w = slope_w_factor[:, :, None] * orographic_factor
        
        # Set wind to zero at ground level (z = 0)
        if self.z_levels[0] == 0:
            self.wind_u[:, :, 0] = 0.0
            self.wind_v[:, :, 0] = 0.0
            self.wind_w[:, :, 0] = 0.0
        
        # Updraft in active fire regions
        if active_fire_time is not None and heat_flux is not None:
            for k in range(self.nz):
                if self.z_levels[k] <= QuicFireConstants.FUEL_HEIGHTS['canopy'] and self.z_levels[k] > 0:
                    fire_mask_k = active_fire_time[:, :, k] > 0
                    if np.any(fire_mask_k):
                        heat_flux_sum = np.sum(heat_flux[fire_mask_k])
                        if QuicFireConstants.HEAT_FLUX > 0:
                            updraft_speed = QuicFireConstants.WIND_UPDRAFT_FACTOR * (heat_flux_sum / QuicFireConstants.HEAT_FLUX)
                            self.wind_w[:, :, k][fire_mask_k] += updraft_speed
                        else:
                            print("Warning: constants.HEAT_FLUX is zero, cannot calculate updraft. Updraft set to 0.0.")
        
        # Clip wind components
        self.wind_u = np.clip(self.wind_u, -2 * wind_speed_ref, 2 * wind_speed_ref)
        self.wind_v = np.clip(self.wind_v, -2 * wind_speed_ref, 2 * wind_speed_ref)
        self.wind_w = np.clip(self.wind_w, 0, 2 * wind_speed_ref)
        
        print(f"✓ Wind field initialized at t={t:.1f}s: "
              f"speed={effective_speed:.2f} m/s, direction={np.degrees(effective_theta):.1f}°, "
              f"max u={np.max(self.wind_u):.2f}, v={np.max(self.wind_v):.2f}, w={np.max(self.wind_w):.2f} m/s")

    def update(self, wind_speed: Optional[float] = None, wind_direction: Optional[float] = None, 
               t: float = 0.0, active_fire_time: Optional[np.ndarray] = None, heat_flux: Optional[np.ndarray] = None) -> None:
        """Update wind field during simulation based on user input or dynamic changes."""
        if self._is_closed:
            print("Warning: WindFieldManager is closed.")
            return None
        
        self._global_wind_avg = None
        wind_speed_to_use = wind_speed if wind_speed is not None else QuicFireConstants.U_REF
        wind_direction_to_use = wind_direction if wind_direction is not None else 0.0
        self.initialize(wind_speed_to_use, wind_direction_to_use, t, active_fire_time, heat_flux)

    def calculate_average_wind(self, active_fire_time: np.ndarray) -> Tuple[float, float, float]:
        """Calculate average wind components."""
        if self._is_closed:
            print("Warning: WindFieldManager is closed.")
            return None
        
        active_fires = active_fire_time > 0
        if np.any(active_fires):
            return (np.mean(self.wind_u[active_fires]), 
                    np.mean(self.wind_v[active_fires]), 
                    np.mean(self.wind_w[active_fires]))
        if self._global_wind_avg is None:
            self._global_wind_avg = (np.mean(self.wind_u), np.mean(self.wind_v), np.mean(self.wind_w))
        return self._global_wind_avg
    
    def get_wind_field(self) -> dict:
        """Get wind field data for visualization or analysis."""
        if self._is_closed:
            print("Warning: WindFieldManager is closed.")
            return None
        
        wind_field_size = self.wind_u.nbytes / (1024 * 1024)
        if wind_field_size > QuicFireConstants.WIND_FIELD_SIZE_LIMIT:
            print(f"Warning: Wind field size {wind_field_size:.1f} MB, which exceeds limit {QuicFireConstants.WIND_FIELD_SIZE_LIMIT:.1f} MB. Consider subsampling for visualization.")
        return {
            'u': self.wind_u.copy(),
            'v': self.wind_v.copy(),
            'w': self.wind_w.copy()
        }

    def generate_rand_wind_updates(self, num_updates: int, total_time: int) -> List[Dict]:
        """Generate realistic wind updates with varying speed and direction."""
        wind_updates = []
        
        initial_speed = self.rng.uniform(0.5, 2.0)
        initial_direction = self.rng.uniform(0, 360)
        
        wind_updates.append({
            'time': 0,
            'speed': round(initial_speed, 1),
            'direction': round(initial_direction, 1)
        })
        
        for i in range(1, num_updates):
            time = int((i / (num_updates - 1)) * total_time)
            prev_speed = wind_updates[-1]['speed']
            prev_direction = wind_updates[-1]['direction']
            
            speed_change = self.rng.uniform(-1.0, 1.5)
            new_speed = max(0.2, min(8.0, prev_speed + speed_change))
            
            if self.rng.uniform(0, 1) < 0.2:
                direction_change = self.rng.uniform(-180, 180)
            else:
                direction_change = self.rng.uniform(-45, 45)
            
            new_direction = (prev_direction + direction_change) % 360
            
            wind_updates.append({
                'time': time,
                'speed': round(new_speed, 1),
                'direction': round(new_direction, 1)
            })
        
        return wind_updates
    
    def close(self):
        """Cleans up resources held by the QuicFireWindFieldManager."""
        if not self._is_closed:
            print("Closing QuicFireWindFieldManager instance...")
            self.wind_u = None
            self.wind_v = None
            self.wind_w = None
            self.fuel_density = None
            self.height_map = None
            self.grad_x = None
            self.grad_y = None
            self.slope_magnitude = None
            self.z_levels = None
            self.rng = None
            self._is_closed = True
            print("QuicFireWindFieldManager closed successfully.")
        else:
            print("QuicFireWindFieldManager is already closed.")

    def visualize_wind_field(self, subsample_factor: int = 5, plot_terrain: bool = True, 
                             title: str = "3D Wind Field", vector_scale: float = 1.0,
                             color_by_speed: bool = True, z_thinning_factor: float = 0.5) -> None:
        """
        Visualizes the 3D wind field using matplotlib's quiver plot with improved clarity.

        Args:
            subsample_factor: Controls the density of visualized vectors. Higher values mean fewer vectors.
            plot_terrain: If True, the terrain height map will be plotted as a surface.
            title: Title of the plot.
            vector_scale: Multiplier for the length of the wind vectors. Adjust to make vectors longer/shorter.
            color_by_speed: If True, vectors will be colored based on their magnitude (wind speed).
            z_thinning_factor: Additional thinning factor applied to the z-dimension.
                               Higher values mean more aggressive thinning vertically.
        """
        if self._is_closed:
            print("Warning: WindFieldManager is closed.")
            return None
        
        if self.nx == 0 or self.ny == 0 or self.nz == 0:
            print("Cannot visualize an empty wind field.")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        x_coords = np.arange(0, self.nx * self.dx, self.dx)
        y_coords = np.arange(0, self.ny * self.dy, self.dy)
        X, Y, Z_levels = np.meshgrid(x_coords, y_coords, self.z_levels, indexing='ij')

        # Apply z_thinning_factor
        z_step = max(1, int(subsample_factor * z_thinning_factor))
        step = subsample_factor

        X_sub = X[::step, ::step, ::z_step]
        Y_sub = Y[::step, ::step, ::z_step]
        Z_sub = Z_levels[::step, ::step, ::z_step]
        U_sub = self.wind_u[::step, ::step, ::z_step]
        V_sub = self.wind_v[::step, ::step, ::z_step]
        W_sub = self.wind_w[::step, ::step, ::z_step]

        # Calculate wind magnitude for coloring and scaling
        wind_magnitude = np.sqrt(U_sub**2 + V_sub**2 + W_sub**2)
        
        # Normalize vector lengths for better visualization
        # The 'length' argument in quiver refers to the data units, not screen units.
        # We want to scale relative to the actual magnitude.
        # A good default length might be related to the grid spacing.
        base_vector_length = self.dx * step * 0.5 * vector_scale 

        if color_by_speed:
            # Flatten arrays for colormapping
            scatter = ax.quiver(X_sub.flatten(), Y_sub.flatten(), Z_sub.flatten(),
                                U_sub.flatten(), V_sub.flatten(), W_sub.flatten(),
                                length=base_vector_length,
                                normalize=False,
                                arrow_length_ratio=0.3,
                                cmap='viridis',  # Use viridis colormap for wind speed
                                colors=plt.cm.viridis(wind_magnitude.flatten() / np.max(wind_magnitude)),
                                alpha=0.7)
            fig.colorbar(scatter, shrink=0.5, aspect=5, label='Wind Speed (m/s)')
        else:
            ax.quiver(X_sub, Y_sub, Z_sub, U_sub, V_sub, W_sub, 
                      length=base_vector_length,
                      normalize=False,
                      arrow_length_ratio=0.3,
                      color='blue', alpha=0.7)

        if plot_terrain and self.height_map is not None:
            X_terrain, Y_terrain = np.meshgrid(x_coords, y_coords, indexing='ij')
            ax.plot_surface(X_terrain, Y_terrain, self.height_map, 
                            cmap='terrain', alpha=0.6, linewidth=0)
            ax.set_zlim(0, max(np.max(self.z_levels), np.max(self.height_map) + 20))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        plt.show()

    def visualize_2d_slices(self,
                       slice_type: str = 'xy',
                       slice_idx: int = 0,
                       subsample_factor: int = 3,
                       plot_terrain_contour: bool = True,
                       title: str = "2D Wind Field Slice",
                       fixed_vector_length: Optional[float] = None) -> None:
        """
        Visualizes 2D slices of the wind field using matplotlib's quiver and contour plots,
        with vectors scaled to a uniform small size and color-coded by velocity.

        Args:
            slice_type: Type of slice ('xy' for horizontal, 'xz' for vertical along x, 'yz' for vertical along y).
            slice_idx: Index of the slice. For 'xy', it's the z-level index. For 'xz', it's the y-index. For 'yz', it's the x-index.
            subsample_factor: Controls the density of visualized vectors in 2D (must be positive).
            plot_terrain_contour: If True, adds contour lines or terrain profile to the plot.
            title: Title of the plot.
            fixed_vector_length: The fixed length for all vectors in data units. If None, a default small length is calculated.
        """
        if self._is_closed:
            print("Warning: WindFieldManager is closed.")
            return

        if subsample_factor < 1:
            print("Error: subsample_factor must be positive.")
            return

        if fixed_vector_length is not None and fixed_vector_length <= 0:
            print("Error: fixed_vector_length must be positive if provided.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Initialize variables for plotting
        X_sub, Y_sub, Z_sub = None, None, None
        U_sub, V_sub, W_sub = None, None, None
        magnitude = None

        if slice_type == 'xy':
            if not (0 <= slice_idx < self.nz):
                print(f"Error: z-level index {slice_idx} out of bounds for nz={self.nz}.")
                plt.close(fig)
                return

            U_slice = self.wind_u[:, :, slice_idx]
            V_slice = self.wind_v[:, :, slice_idx]

            x_coords = np.arange(0, self.nx * self.dx, self.dx)
            y_coords = np.arange(0, self.ny * self.dy, self.dy)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

            # Subsample for quiver plot
            step = subsample_factor
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            U_sub = U_slice[::step, ::step]
            V_sub = V_slice[::step, ::step]

            # Calculate magnitude for coloring
            magnitude = np.sqrt(U_sub**2 + V_sub**2)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{title} (XY-plane at Z={self.z_levels[slice_idx]:.2f}m)')

            if plot_terrain_contour:
                contour = ax.contour(X, Y, self.height_map, levels=10, colors='gray', linestyles='--', alpha=0.7)
                ax.clabel(contour, inline=True, fontsize=8)
                ax.set_title(f'{title} (XY-plane at Z={self.z_levels[slice_idx]:.2f}m) with Terrain Contours')

        elif slice_type == 'xz':
            if not (0 <= slice_idx < self.ny):
                print(f"Error: y-index {slice_idx} out of bounds for ny={self.ny}.")
                plt.close(fig)
                return

            U_slice = self.wind_u[:, slice_idx, :]
            W_slice = self.wind_w[:, slice_idx, :]

            x_coords = np.arange(0, self.nx * self.dx, self.dx)
            X, Z = np.meshgrid(x_coords, self.z_levels, indexing='ij')

            # Subsample for quiver plot
            step = subsample_factor
            X_sub = X[::step, ::step]
            Z_sub = Z[::step, ::step]
            U_sub = U_slice[::step, ::step]
            W_sub = W_slice[::step, ::step]

            # Calculate magnitude for coloring
            magnitude = np.sqrt(U_sub**2 + W_sub**2)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(f'{title} (XZ-plane at Y={slice_idx * self.dy:.2f}m)')

            if plot_terrain_contour:
                ax.plot(x_coords, self.height_map[:, slice_idx], color='brown', linestyle='-', linewidth=2, label='Terrain Profile')
                ax.fill_between(x_coords, 0, self.height_map[:, slice_idx], color='sienna', alpha=0.3)
                ax.legend()

        elif slice_type == 'yz':
            if not (0 <= slice_idx < self.nx):
                print(f"Error: x-index {slice_idx} out of bounds for nx={self.nx}.")
                plt.close(fig)
                return

            V_slice = self.wind_v[slice_idx, :, :]
            W_slice = self.wind_w[slice_idx, :, :]

            y_coords = np.arange(0, self.ny * self.dy, self.dy)
            Y, Z = np.meshgrid(y_coords, self.z_levels, indexing='ij')

            # Subsample for quiver plot
            step = subsample_factor
            Y_sub = Y[::step, ::step]
            Z_sub = Z[::step, ::step]
            V_sub = V_slice[::step, ::step]
            W_sub = W_slice[::step, ::step]

            # Calculate magnitude for coloring
            magnitude = np.sqrt(V_sub**2 + W_sub**2)

            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(f'{title} (YZ-plane at X={slice_idx * self.dx:.2f}m)')

            if plot_terrain_contour:
                ax.plot(y_coords, self.height_map[slice_idx, :], color='brown', linestyle='-', linewidth=2, label='Terrain Profile')
                ax.fill_between(y_coords, 0, self.height_map[slice_idx, :], color='sienna', alpha=0.3)
                ax.legend()

        else:
            print("Invalid slice_type. Choose 'xy', 'xz', or 'yz'.")
            plt.close(fig)
            return

        # Set default fixed_vector_length based on grid spacing
        if fixed_vector_length is None:
            if slice_type == 'xy':
                fixed_vector_length = max(self.dx, self.dy) * subsample_factor #* 0.2
            else:
                # Use median z-level spacing for robustness
                z_spacing = np.median(np.diff(self.z_levels)) if len(self.z_levels) > 1 else self.z_levels[0]
                fixed_vector_length = max(self.dx, self.dy, z_spacing) * subsample_factor # * 0.2

        # Plot quiver with consistent scaling
        if slice_type == 'xy':
            qv = ax.quiver(X_sub, Y_sub, U_sub, V_sub,
                        magnitude,  # Color by magnitude
                        cmap='viridis',
                        scale_units='xy',
                        scale=np.max(magnitude) / fixed_vector_length * 1.5 if np.max(magnitude) > 0 else 1.0,
                        width=0.005,
                        headwidth=3, headlength=5, headaxislength=4,
                        alpha=0.8)
        else:
            # For 'xz' and 'yz', use appropriate components
            qv = ax.quiver(X_sub if slice_type == 'xz' else Y_sub, Z_sub,
                        U_sub if slice_type == 'xz' else V_sub, W_sub,
                        magnitude,  # Color by magnitude
                        cmap='viridis',
                        scale_units='xy',
                        scale=np.max(magnitude) / fixed_vector_length * 1.5 if np.max(magnitude) > 0 else 1.0,
                        width=0.005,
                        headwidth=3, headlength=5, headaxislength=4,
                        alpha=0.8)
            ax.set_ylim(0, np.max(self.z_levels) + 10)

        cbar = fig.colorbar(qv, ax=ax, shrink=0.75, aspect=20, label='Wind Speed (m/s)')
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
