import numpy as np
from typing import List, Tuple, Optional, Dict
import math
from QuicFire.QuicFireConstants import QuicFireConstants

class IgnitionLine:
    """Represents a single ignition line with properties."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 width: float = 2.0, ignition_energy_factor: float = 10.0, 
                 moisture: float = 0.01, eps: float = 2.0, 
                 depletion: float = 0.8, temp_delta: float = 100.0):
        """
        Initialize an ignition line.
        
        Args:
            x1, y1: Start coordinates (in grid indices)
            x2, y2: End coordinates (in grid indices)
            width: Line width (in meters)
            ignition_energy_factor: Ignition multiplier (>= 1.0 for fire to start)
            moisture: Initial moisture fraction
            eps: Initial ember particles
            depletion: Moisture depletion fraction
            temp_delta: Temperature increase above ambient (K)
        """
        assert ignition_energy_factor > 1.0, "ignition_energy_factor must be > 1.0 to ignite fire"

        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.width = width
        self.ignition_energy_factor = ignition_energy_factor

        self.moisture = moisture
        self.eps = eps
        self.depletion = depletion
        self.temp_delta = temp_delta
        
        # Calculate line properties
        self.length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.angle = math.atan2(y2 - y1, x2 - x1)
        
    def get_line_cells(self, dx: float, dy: float, nx: int, ny: int, burn_box_coords: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """Get all grid cells that this line passes through, constrained to burn box if provided."""
        cells = []
        
        # Convert coordinates to grid indices
        i1, j1 = int(self.x1), int(self.y1)
        i2, j2 = int(self.x2), int(self.y2)
        
        # Use Bresenham's line algorithm for the main line
        line_cells = self._bresenham_line(i1, j1, i2, j2)
        
        # Define burn box boundaries for cell filtering
        if burn_box_coords:
            min_x, min_y, max_x, max_y = burn_box_coords
        else:
            min_x, min_y = 0, 0
            max_x, max_y = nx - 1, ny - 1
        
        # Add width by expanding perpendicular to the line
        if self.width > min(dx, dy):
            width_cells = int(self.width / min(dx, dy))
            expanded_cells = set()
            
            for i, j in line_cells:
                # Add cells perpendicular to the line direction
                perp_angle = self.angle + math.pi / 2
                for w in range(-width_cells // 2, width_cells // 2 + 1):
                    offset_i = int(w * math.cos(perp_angle))
                    offset_j = int(w * math.sin(perp_angle))
                    new_i, new_j = i + offset_i, j + offset_j
                    
                    # Only include cells within burn box or grid boundaries
                    if min_x <= new_i <= max_x and min_y <= new_j <= max_y:
                        expanded_cells.add((new_i, new_j))
            
            cells = list(expanded_cells)
        else:
            cells = [(i, j) for i, j in line_cells if min_x <= i <= max_x and min_y <= j <= max_y]
        
        return cells
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm to get all cells along a line."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points

class QuicFireIgnitionManager:
    """Manages multiple ignition lines and applies them to the simulation."""
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.ignition_lines: List[IgnitionLine] = []
        self._is_closed = False
    
    def add_line(self, x1: float, y1: float, x2: float, y2: float, **kwargs) -> None:
        if self._is_closed:
            print("Warning: Cannot add line to a closed IgnitionManager.")
            return
        line = IgnitionLine(x1, y1, x2, y2, **kwargs)
        self.ignition_lines.append(line)
        
    def add_lines_from_coords(self, line_coords: List[Tuple[float, float, float, float]], **kwargs) -> None:
        if self._is_closed:
            print("Warning: Cannot add lines to a closed IgnitionManager.")
            return
        for x1, y1, x2, y2 in line_coords:
            self.add_line(x1, y1, x2, y2, **kwargs)
    
    def clear_lines(self) -> None:
        if self._is_closed:
            print("Warning: Cannot clear lines on a closed IgnitionManager.")
            return
        self.ignition_lines.clear()
    
    def get_line_count(self) -> int:
        return len(self.ignition_lines)
    
    def get_total_length(self) -> float:
        return sum(line.length for line in self.ignition_lines)
    
    def apply_ignition(self, simulation, burn_box_coords: Optional[List[int]] = None) -> int:
        """Apply all ignition lines to the simulation at the terrain height, constrained to burn box."""
        if self._is_closed:
            print("Warning: Cannot apply ignition from a closed IgnitionManager.")
            return 0

        ignited_cells = 0

        # Validate simulation attributes
        if not hasattr(simulation, 'height_map') or simulation.height_map.shape != (simulation.nx, simulation.ny):
            raise ValueError(f"height_map must have shape (nx, ny) = ({simulation.nx}, {simulation.ny})")
        if np.any(simulation.height_map < 0) or np.any(simulation.height_map > simulation.z_levels.max()):
            raise ValueError(f"height_map values must be in [0, {simulation.z_levels.max()}]")

        for line in self.ignition_lines:
            cells = line.get_line_cells(simulation.dx, simulation.dy, simulation.nx, simulation.ny, burn_box_coords)
            
            for i, j in cells:
                terrain_height = simulation.height_map[i, j]
                k_ground = np.searchsorted(simulation.z_levels, terrain_height, side='right')
                k_ground = min(k_ground, simulation.nz - 1)

                if k_ground >= simulation.nz:
                    continue
                
                if simulation.fuel_density[i, j, k_ground] > QuicFireConstants.FUEL_DENSITY_THRESHOLD:
                    simulation.moisture_fraction[i, j, k_ground] = line.moisture * line.ignition_energy_factor
                    simulation.EPs[i, j, k_ground] = line.eps * line.ignition_energy_factor
                    simulation.moisture_depletion_fraction[i, j, k_ground] = line.depletion
                    simulation.temperature[i, j, k_ground] = (QuicFireConstants.T_AMBIENT + 
                                                             line.temp_delta * line.ignition_energy_factor)
                    ignited_cells += 1
                
        return ignited_cells

    def generate_rand_ignition_lines(self, nx: int, ny: int, dx: float = 2.0, dy: float = 2.0, num_lines: int = 2, burn_box_coords: Optional[List[int]] = None, inner_buffer: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Generate realistic ignition line coordinates strictly within the burn box or grid boundaries.
        
        Args:
            nx, ny: Grid dimensions
            num_lines: Number of ignition lines to generate
            burn_box_coords: [x_min, y_min, x_max, y_max] in grid indices
            dx, dy: Grid spacing in meters
        """
        lines = []
        width_buffer = max(1, int(3.0 / min(dx, dy)))
        if burn_box_coords:
            min_x = max(0, burn_box_coords[0] + width_buffer + inner_buffer)
            min_y = max(0, burn_box_coords[1] + width_buffer + inner_buffer)
            max_x = min(nx - 1, burn_box_coords[2] - width_buffer - inner_buffer)
            max_y = min(ny - 1, burn_box_coords[3] - width_buffer - inner_buffer)
            if min_x > max_x or min_y > max_y:
                print("Warning: Burn box too small after buffers. Returning empty list.")
                return []
        else:
            margin = max(2, min(nx // 10, ny // 10))
            min_x, min_y = margin, margin
            max_x, max_y = nx - 1 - margin, ny - 1 - margin

        effective_width = max_x - min_x
        effective_height = max_y - min_y

        for _ in range(num_lines):
            line_type = self.rng.choice(['horizontal', 'vertical', 'diagonal', 'curved'])
            
            if line_type == 'horizontal':
                y = self.rng.integers(min_y, max_y + 1)
                x1 = self.rng.integers(min_x, min_x + effective_width // 2 + 1)
                x2 = self.rng.integers(x1, max_x + 1)
                lines.append((x1, y, x2, y))
                
            elif line_type == 'vertical':
                x = self.rng.integers(min_x, max_x + 1)
                y1 = self.rng.integers(min_y, min_y + effective_height // 2 + 1)
                y2 = self.rng.integers(y1, max_y + 1)
                lines.append((x, y1, x, y2))
                
            elif line_type == 'diagonal':
                max_length = min(effective_width, effective_height) // 2
                if max_length < 5:
                    max_length = min(effective_width, effective_height)
                length = self.rng.integers(5, max_length + 1)
                
                x1 = self.rng.integers(min_x, max_x - length + 1)
                y1 = self.rng.integers(min_y, max_y - length + 1)
                x_dir = self.rng.choice([-1, 1])
                y_dir = self.rng.choice([-1, 1])
                
                x2 = x1 + x_dir * length
                y2 = y1 + y_dir * length
                x2 = max(min_x, min(max_x, x2))
                y2 = max(min_y, min(max_y, y2))
                
                lines.append((x1, y1, x2, y2))
                
            else:  # curved
                max_radius = min(effective_width, effective_height) // 3
                if max_radius < 5:
                    max_radius = min(effective_width, effective_height) // 2
                radius = self.rng.integers(5, max_radius + 1)
                
                center_x = self.rng.integers(min_x + radius, max_x - radius + 1)
                center_y = self.rng.integers(min_y + radius, max_y - radius + 1)
                
                start_angle = self.rng.uniform(0, 2 * math.pi)
                end_angle = start_angle + self.rng.uniform(math.pi/8, math.pi/4)
                
                x1 = center_x + radius * math.cos(start_angle)
                y1 = center_y + radius * math.sin(start_angle)
                x2 = center_x + radius * math.cos(end_angle)
                y2 = center_y + radius * math.sin(end_angle)
                
                # Convert to grid indices and clamp
                x1 = max(min_x, min(max_x, int(x1)))
                y1 = max(min_y, min(max_y, int(y1)))
                x2 = max(min_x, min(max_x, int(x2)))
                y2 = max(min_y, min(max_y, int(y2)))
                
                lines.append((x1, y1, x2, y2))
        
        return lines
    
    def generate_rand_ignition_params(self) -> Dict:
        """Generate realistic ignition parameters with randomization."""
        return {
            'width': self.rng.uniform(1.5, 3.0),
            'ignition_energy_factor': self.rng.uniform(1.0, 2.0),
            'moisture': self.rng.uniform(0.005, 0.02),
            'eps': self.rng.uniform(1.5, 2.5),
            'depletion': self.rng.uniform(0.1, 0.3),
            'temp_delta': self.rng.uniform(80.0, 120.0)
        }
    
    def get_ignition_summary(self) -> Dict:
        """Get a summary of all ignition lines."""
        if not self.ignition_lines:
            return {"lines": 0, "total_length": 0.0}
        
        summary = {
            "lines": len(self.ignition_lines),
            "total_length": self.get_total_length(),
            "line_details": []
        }
        
        for i, line in enumerate(self.ignition_lines):
            summary["line_details"].append({
                "id": i,
                "start": (line.x1, line.y1),
                "end": (line.x2, line.y2),
                "length": line.length,
                "width": line.width,
                "ignition_energy_factor": line.ignition_energy_factor
            })
        
        return summary

    def close(self):
        """Cleans up resources held by the QuicFireIgnitionManager."""
        if not self._is_closed:
            print("Closing QuicFireIgnitionManager instance...")
            self.ignition_lines.clear()
            self.ignition_lines = None
            self._is_closed = True
            print("QuicFireIgnitionManager closed successfully.")
        else:
            print("QuicFireIgnitionManager is already closed.")