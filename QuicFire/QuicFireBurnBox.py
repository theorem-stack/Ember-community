import numpy as np
import warnings
from skimage.transform import resize
from typing import Tuple, List, Optional, Dict
from QuicFire.QuicFireConstants import QuicFireConstants

class QuicFireBurnBox:
    """A class to define a rectangular burn box for prescribed burns in a wildfire simulation.
    
    The burn box is a target area where fires are allowed, with the agent's goal to minimize
    fire spread outside this area. Supports user-defined coordinates or random generation
    with a seed for reproducibility. Ensures padding is prioritized and allows area constraints
    in either grid cells or fractions of the effective grid area, accounting for ignition line buffers.
    """
    
    def __init__(self, rng: np.random.Generator, nx: int, ny: int, coords: Optional[List[int]] = None, 
                 min_area: float = 0.2, max_area: float = 0.8, area_unit: str = 'fraction', 
                 dx: float = 2.0, dy: float = 2.0, ignition_inner_buffer: int = 5):
        """
        Initialize the burn box with padding priority and flexible area constraints.
        
        Args:
            rng (np.random.Generator): Random number generator for reproducibility.
            nx (int): Simulation grid width.
            ny (int): Simulation grid height.
            coords (list, optional): [x_min, y_min, x_max, y_max] in simulation grid coordinates.
                                     If provided, adjusted to respect padding and ignition buffers.
            min_area (float): Minimum area constraint (in grid cells or fraction of effective area).
            max_area (float): Maximum area constraint (in grid cells or fraction of effective area).
            area_unit (str): Unit of area constraints, 'fraction' or 'cells'. Default is 'fraction'.
            dx (float): Grid spacing in x-direction (meters). Default is 2.0.
            dy (float): Grid spacing in y-direction (meters). Default is 2.0.
            ignition_inner_buffer (int): Inner buffer for ignition lines (grid cells). Default is 5.
        """
        self.rng = rng
        self.nx = nx
        self.ny = ny
        self.min_area = min_area
        self.max_area = max_area
        self.area_unit = area_unit
        self.dx = dx
        self.dy = dy
        self.ignition_inner_buffer = ignition_inner_buffer

        # Validate area_unit
        if self.area_unit not in ['fraction', 'cells']:
            raise ValueError("area_unit must be 'fraction' or 'cells'")

        # Calculate ignition line buffers
        self.width_buffer = max(1, int(3.0 / min(self.dx, self.dy)))
        self.total_ignition_buffer = self.width_buffer + self.ignition_inner_buffer

        # Validate grid size against padding
        min_grid_size = 2 * QuicFireConstants.BURN_BOX_PADDING + 2 * self.total_ignition_buffer + 10
        if self.nx <= min_grid_size or self.ny <= min_grid_size:
            raise ValueError(
                f"Grid dimensions ({self.nx}x{self.ny}) are too small for padding "
                f"({QuicFireConstants.BURN_BOX_PADDING}) and ignition buffers "
                f"({self.total_ignition_buffer}). Minimum required: {min_grid_size}x{min_grid_size}."
            )
        
        # Calculate effective grid dimensions (after padding)
        self.effective_nx = self.nx - 2 * QuicFireConstants.BURN_BOX_PADDING
        self.effective_ny = self.ny - 2 * QuicFireConstants.BURN_BOX_PADDING

        # Calculate minimum area for ignition lines (22x22 grid cells after buffers)
        min_ignition_area = (2 * self.total_ignition_buffer + 10) ** 2

        # Set area constraints
        if self.area_unit == 'fraction':
            effective_grid_area = self.effective_nx * self.effective_ny
            self.min_effective_area = max(self.min_area * effective_grid_area, min_ignition_area)
            self.max_effective_area = min(self.max_area * effective_grid_area, effective_grid_area)
        else:  # 'cells'
            self.min_effective_area = max(self.min_area, min_ignition_area)
            self.max_effective_area = self.max_area

        # Validate area constraints
        max_possible_area = self.effective_nx * self.effective_ny
        if self.min_effective_area > max_possible_area:
            raise ValueError(
                f"Minimum area {self.min_effective_area} ({self.area_unit}) exceeds effective grid area {max_possible_area}"
            )
        if self.max_effective_area > max_possible_area:
            warnings.warn(
                f"Maximum area {self.max_area} ({self.area_unit}) exceeds effective grid area {max_possible_area}, "
                f"setting to {max_possible_area}"
            )
            self.max_effective_area = max_possible_area

        if coords is not None:
            self.coords = self._validate_and_adjust_coords(coords)
        else:
            self.coords = self._generate_random_coords()
        
        self.sim_mask = self._create_sim_mask()
    
    def _validate_and_adjust_coords(self, coords: List[int]) -> List[int]:
        """
        Validate and adjust user-provided coordinates to respect padding, ignition buffers, and area constraints.
        
        Args:
            coords (List[int]): [x_min, y_min, x_max, y_max].
        
        Returns:
            List[int]: Adjusted coordinates [x_min, y_min, x_max, y_max].
        
        Raises:
            ValueError: If coordinates are invalid or violate constraints after adjustment.
        """
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError("Coords must be a list or tuple of [x_min, y_min, x_max, y_max]")
        
        x_min, y_min, x_max, y_max = map(int, coords)
        
        # Check initial bounds
        if not (0 <= x_min <= x_max < self.nx and 0 <= y_min <= y_max < self.ny):
            raise ValueError(f"Coordinates {coords} are out of grid bounds {self.nx}x{self.ny}")

        # Adjust to respect padding
        adjusted_x_min = max(x_min, QuicFireConstants.BURN_BOX_PADDING)
        adjusted_y_min = max(y_min, QuicFireConstants.BURN_BOX_PADDING)
        adjusted_x_max = min(x_max, self.nx - 1 - QuicFireConstants.BURN_BOX_PADDING)
        adjusted_y_max = min(y_max, self.ny - 1 - QuicFireConstants.BURN_BOX_PADDING)

        # Check if adjustment invalidates the box
        if adjusted_x_min > adjusted_x_max or adjusted_y_min > adjusted_y_max:
            raise ValueError(
                f"Coordinates {coords} cannot form a valid box after applying padding "
                f"{QuicFireConstants.BURN_BOX_PADDING}"
            )

        # Validate effective area for ignition lines
        effective_width = adjusted_x_max - adjusted_x_min - 2 * self.total_ignition_buffer
        effective_height = adjusted_y_max - adjusted_y_min - 2 * self.total_ignition_buffer
        if effective_width < 10 or effective_height < 10:
            raise ValueError(
                f"Adjusted coordinates {coords} yield effective area ({effective_width}x{effective_height}) "
                f"too small for ignition lines after buffers ({self.total_ignition_buffer})"
            )

        # Validate area constraints
        area = (adjusted_x_max - adjusted_x_min + 1) * (adjusted_y_max - adjusted_y_min + 1)
        if area < self.min_effective_area:
            raise ValueError(
                f"Adjusted area {area} is below minimum {self.min_effective_area} ({self.area_unit})"
            )
        if area > self.max_effective_area:
            warnings.warn(
                f"Adjusted area {area} exceeds maximum {self.max_effective_area} ({self.area_unit})"
            )

        adjusted_coords = [adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max]
        if adjusted_coords != list(coords):
            warnings.warn(
                f"Coordinates adjusted from {coords} to {adjusted_coords} "
                f"to respect padding {QuicFireConstants.BURN_BOX_PADDING}"
            )

        return adjusted_coords
    
    def _generate_random_coords(self) -> List[int]:
        """
        Generate random burn box coordinates within the effective grid, respecting padding, 
        ignition buffers, and area constraints.
        
        Returns:
            List[int]: [x_min, y_min, x_max, y_max].
        """
        # Minimum dimensions to support ignition lines after buffers
        min_width = 2 * self.total_ignition_buffer + 10
        min_height = 2 * self.total_ignition_buffer + 10
        min_area = max(self.min_effective_area, min_width * min_height)

        while True:
            # Randomly select width and height within effective grid bounds
            width = self.rng.integers(max(min_width, 1), self.effective_nx + 1)
            height = self.rng.integers(max(min_height, 1), self.effective_ny + 1)
            area = width * height
            if min_area <= area <= self.max_effective_area:
                break

        # Place the box within the padded region
        x_min = self.rng.integers(
            QuicFireConstants.BURN_BOX_PADDING,
            self.nx - width - QuicFireConstants.BURN_BOX_PADDING + 1
        )
        y_min = self.rng.integers(
            QuicFireConstants.BURN_BOX_PADDING,
            self.ny - height - QuicFireConstants.BURN_BOX_PADDING + 1
        )
        x_max = x_min + width - 1
        y_max = y_min + height - 1
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def _create_sim_mask(self) -> np.ndarray:
        """
        Create a boolean mask for the burn box at simulation grid resolution.
        
        Returns:
            np.ndarray: Boolean mask (True inside, False outside).
        """
        mask = np.zeros((self.nx, self.ny), dtype=np.bool_)
        x_min, y_min, x_max, y_max = self.coords
        mask[x_min:x_max + 1, y_min:y_max + 1] = True
        return mask

    def create_agent_obs_mask(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a down-sampled boolean mask of shape (obs_nx, obs_ny) from the simulation mask.
        
        Args:
            target_shape (Tuple[int, int]): Target shape (obs_nx, obs_ny) of the observation mask.
        
        Returns:
            np.ndarray: Down-sampled boolean mask (True inside burn box, False outside).
        
        Raises:
            ValueError: If target dimensions are invalid or incompatible with simulation mask.
        """
        obs_nx, obs_ny = target_shape

        if not (isinstance(obs_nx, int) and isinstance(obs_ny, int) and obs_nx > 0 and obs_ny > 0):
            raise ValueError(f"Invalid target dimensions: obs_nx={obs_nx}, obs_ny={obs_ny} must be positive integers")
        
        if obs_nx > self.nx or obs_ny > self.ny:
            raise ValueError(
                f"Observation mask dimensions ({obs_nx}x{obs_ny}) exceed simulation grid ({self.nx}x{self.ny})"
            )

        # Resize the simulation mask to the target shape
        downsampled_mask = resize(
            self.sim_mask,
            output_shape=(obs_nx, obs_ny),
            mode='constant',
            cval=0.0,
            anti_aliasing=False,
            preserve_range=True
        )
        
        # Convert to boolean
        binary_mask = (downsampled_mask >= 0.5).astype(np.bool_)
        
        if binary_mask.shape != (obs_nx, obs_ny):
            raise ValueError(
                f"Down-sampled mask shape {binary_mask.shape} does not match requested ({obs_nx}x{obs_ny})"
            )

        return binary_mask

    def get_sim_mask(self) -> np.ndarray:
        """Return the simulation-resolution burn box mask."""
        return self.sim_mask

    def get_coords(self) -> List[int]:
        """Return the burn box coordinates [x_min, y_min, x_max, y_max]."""
        return self.coords
    
    def to_dict(self) -> Dict:
        """Convert burn box to a dictionary for serialization."""
        return {
            'nx': self.nx,
            'ny': self.ny,
            'coords': self.coords,
            'min_area': self.min_area,
            'max_area': self.max_area,
            'area_unit': self.area_unit,
            'burn_box_padding': QuicFireConstants.BURN_BOX_PADDING,
            'dx': self.dx,
            'dy': self.dy,
            'ignition_inner_buffer': self.ignition_inner_buffer
        }