import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
from typing import Dict, Any
from matplotlib.patches import Rectangle
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireBurnBox import QuicFireBurnBox

def visualize_fire_movement(results: Dict, nx: int, ny: int, dx: float = 2.0, dy: float = 2.0, burn_box: Any = None):
    """
    Visualize fire movement and average mass burn rate over time, with burn box perimeter on every frame.
    Fire intensity is normalized so that the maximum intensity is 1 and all other values are between 0 and 1.

    Args:
        results (dict): Simulation results containing 'time_series' and 'consumption_rate_history'.
        nx (int): Number of grid cells in x-direction.
        ny (int): Number of grid cells in y-direction.
        dx (float): Grid spacing in x-direction (meters). Default is 2.0.
        dy (float): Grid spacing in y-direction (meters). Default is 2.0.
        burn_box (QuicFireBurnBox, optional): Burn box object to visualize boundaries.
    """
    if 'time_series' not in results:
        print("No time series data available.")
        return

    time_series = results['time_series']
    num_frames = len(time_series)
    
    # Extract consumption_rate_history
    if 'consumption_rate_history' not in results:
        print("Warning: consumption_rate_history not found in results. Skipping burn rate plot.")
        consumption_rate_history = np.zeros(num_frames)  # Dummy data for plotting
    else:
        consumption_rate_history = np.array(results['consumption_rate_history'])
    
    extent = [0, nx * dx, 0, ny * dy]  # Extent in meters
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- Fire Movement Plot (Left Subplot) ---
    def get_fire_grid(frame, normalize=True):
        fires = time_series[frame]['data']['fires']
        fire_grid = np.zeros((ny, nx))
        
        if fires:
            for fire in fires:
                position = fire['position']
                intensity = fire['intensity']
                
                x, y, _ = position
                
                grid_i = int(np.round(x / dx))
                grid_j = int(np.round(y / dy))
                
                if 0 <= grid_i < nx and 0 <= grid_j < ny:
                    fire_grid[grid_j, grid_i] += intensity
        
        # Normalize the fire_grid so that the maximum intensity is 1
        if normalize and global_max_intensity > 0:
            fire_grid = fire_grid / global_max_intensity
        
        return fire_grid
    
    # Calculate the global maximum intensity for normalization
    all_intensities = []
    for frame in range(num_frames):
        fire_grid = get_fire_grid(frame, normalize=False)  # Get raw intensities
        if fire_grid.size > 0:
            all_intensities.append(np.max(fire_grid))
            
    global_max_intensity = max(all_intensities) if all_intensities else 1.0 

    # Normalize colormap to 0-1 range
    norm = Normalize(vmin=0, vmax=1)

    fire_grid = get_fire_grid(0)
    im = ax1.imshow(fire_grid, origin='lower', extent=extent, cmap='hot', norm=norm)
    ax1.set_title("Fire Movement with Burn Box")
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Color bar for normalized intensity (0 to 1)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Fire Intensity')
    
    # Initialize burn box patch
    burn_box_patch = None
    if burn_box is not None:
        x_min, y_min, x_max, y_max = burn_box.get_coords()
        rect_x = x_min * dx
        rect_y = y_min * dy
        rect_width = (x_max - x_min + 1) * dx
        rect_height = (y_max - y_min + 1) * dy
        burn_box_patch = Rectangle(
            (rect_x, rect_y), rect_width, rect_height,
            fill=False, edgecolor='white', linewidth=2, linestyle='--'
        )
        ax1.add_patch(burn_box_patch)
    
    # --- Consumption Rate Plot (Right Subplot) ---
    time_points = [ts['time'] for ts in time_series]
    
    if len(consumption_rate_history) >= num_frames:
        line, = ax2.plot(time_points[0], consumption_rate_history[0], 'b-')
        ax2.set_xlim(0, time_points[-1] if time_points else 1)
        ax2.set_ylim(0, np.max(consumption_rate_history) * 1.1 if np.any(consumption_rate_history) else 1)
        ax2.set_title("Average Mass Burn Rate")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Avg. Mass Burn Rate (kg/m²/s)")
        ax2.grid(True)
    else:
        print("Warning: Consumption rate history length does not match number of frames. Skipping plot.")
        ax2.set_visible(False)
    
    plt.tight_layout()

    def update(frame):
        # Update fire movement plot
        fire_grid = get_fire_grid(frame)
        im.set_data(fire_grid)
        ax1.set_title(f"Fire Movement with Burn Box at t={time_series[frame]['time']:.2f}s")
        
        # Clear existing burn box patch and redraw
        new_patch = None
        if burn_box is not None:
            if ax1.patches:
                ax1.patches[-1].remove()
            x_min, y_min, x_max, y_max = burn_box.get_coords()
            rect_x = x_min * dx
            rect_y = y_min * dy
            rect_width = (x_max - x_min + 1) * dx
            rect_height = (y_max - y_min + 1) * dy
            new_patch = Rectangle(
                (rect_x, rect_y), rect_width, rect_height,
                fill=False, edgecolor='white', linewidth=2, linestyle='--'
            )
            ax1.add_patch(new_patch)
        
        # Update consumption rate plot
        if len(consumption_rate_history) >= num_frames:
            current_time_up_to_frame = time_points[:frame+1]
            current_consumption_data = consumption_rate_history[:frame+1]
            line.set_data(current_time_up_to_frame, current_consumption_data)
            
            artists = [im, line]
            if new_patch is not None:
                artists.append(new_patch)
            return artists
        
        artists = [im]
        if new_patch is not None:
            artists.append(new_patch)
        return artists
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)
    plt.show()

def visualize_results(height_map: np.ndarray, fuel_density: np.ndarray, nx: float, ny: float, nz: float, dx: float, dy: float, z_levels: Dict):
        """
        Improved visualization method for generated terrain and 3D fuel density.
        Uses more efficient indexing and options for better clarity and potential interactivity.
        """
        if height_map.shape != (nx, ny):
            raise ValueError(f"Terrain shape {height_map.shape} does not match (nx, ny) = ({nx}, {ny})")
        if fuel_density.shape != (nx, ny, nz):
            raise ValueError(f"Fuel density shape {fuel_density.shape} does not match (nx, ny, nz) = ({nx}, {ny}, {nz})")
        
        fig = plt.figure(figsize=(12, 10)) # Adjusted figure size
        ax = fig.add_subplot(111, projection='3d') # Single subplot for clarity, or integrate into a larger dashboard

        # --- (a) Plot the Terrain Surface ---
        X_grid, Y_grid = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy, indexing='ij')
        ax.plot_surface(X_grid, Y_grid, height_map, cmap='terrain', alpha=0.8, rstride=1, cstride=1, shade=False)
        
        # --- (b) Filter and Plot 3D Fuel Density (more efficiently) ---
        # Use np.where to get indices directly, avoiding full meshgrid.
        # This is more memory efficient for large grids.
        min_density_threshold = QuicFireConstants.RHO_F0 * 0.01 
        
        # Get indices where fuel density is above threshold
        # This directly gives i, j, k arrays
        i_filtered, j_filtered, k_filtered = np.where(fuel_density > min_density_threshold)

        if len(i_filtered) == 0:
            ax.text2D(0.5, 0.5, 'No fuel above threshold', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=16, color='red')
        else:
            # Map indices to actual coordinates
            x_filtered = i_filtered * dx
            y_filtered = j_filtered * dy
            z_filtered = z_levels[k_filtered] # Directly use the z_levels for the k-indices

            density_filtered = fuel_density[i_filtered, j_filtered, k_filtered]
            
            # Normalize density for marker size, potentially using a logarithmic scale for better distinction of lower values
            # If density_filtered has a wide range, linear normalization can make low values invisible.
            norm_density = Normalize(vmin=density_filtered.min(), vmax=density_filtered.max())
            density_normalized_for_size = norm_density(density_filtered)
            
            # Use log scale for size to make lower densities more visible, or a power scale
            s_vals = 10 + 50 * (density_normalized_for_size**0.5) # Power scale for size, adjust exponent as needed

            # Color mapping: Viridis is good, consider 'hot' or 'YlOrRd' for fire-related intuition
            # Use a consistent color mapping (e.g., vmin/vmax) if comparing different plots
            scatter = ax.scatter(x_filtered, y_filtered, z_filtered, 
                                c=density_filtered, cmap='viridis', 
                                s=s_vals, # Use calculated sizes
                                alpha=0.7, # Slightly higher alpha for more visibility
                                edgecolors='none') 
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.05) # Adjusted shrink, aspect, and pad
            cbar.set_label('Fuel Density (kg/m³)', rotation=270, labelpad=20, fontsize=12)
            
        # --- (c) Set consistent Limits and Aspect Ratio ---
        max_possible_z = max(z_levels.max(), np.max(height_map) + 50) # Ensure terrain is within limits
        
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(0, ny * dy)
        ax.set_zlim(0, max_possible_z) 

        # Set aspect ratio to match physical dimensions. Using `set_box_aspect` is good.
        ax.set_box_aspect([nx * dx, ny * dy, max_possible_z]) 
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_title('3D Fuel Density on Terrain', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        
        # --- (d) Camera Angle and Interactivity ---
        # Default view
        ax.view_init(elev=35, azim=-50) # Slightly higher elevation, different azimuth

        # For interactive exploration in Jupyter notebooks, uncomment the following:
        # This allows you to dynamically change elevation and azimuth.
        # def update_view(elev, azim):
        #     ax.view_init(elev=elev, azim=azim)
        #     fig.canvas.draw_idle()
        #
        # elev_slider = widgets.IntSlider(value=35, min=0, max=90, step=5, description='Elevation:')
        # azim_slider = widgets.IntSlider(value=-50, min=-180, max=180, step=5, description='Azimuth:')
        #
        # widgets.interactive(update_view, elev=elev_slider, azim=azim_slider)
        # display(widgets.VBox([elev_slider, azim_slider]))

        plt.tight_layout()
        plt.show()