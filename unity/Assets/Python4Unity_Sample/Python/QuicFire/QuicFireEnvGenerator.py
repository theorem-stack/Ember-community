import numpy as np
# import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from enum import Enum
# from matplotlib.colors import ListedColormap
# import os
# from QuicFire.RNGCollection import RNGCollection
# from PIL import Image
from dataclasses import dataclass
from QuicFire.QuicFireConstants import QuicFireConstants

class VegetationType(Enum):
    WATER = 0
    BARE_SOIL = 1
    GRASS = 2
    SHRUB = 3
    DECIDUOUS_TREE = 4
    CONIFEROUS_TREE = 5
    DEAD_VEGETATION = 6

class BiomeType(Enum):
    DESERT = 0
    GRASSLAND = 1
    TEMPERATE_FOREST = 2
    BOREAL_FOREST = 3
    ALPINE = 4
    WETLAND = 5

@dataclass
class VegetationProperties:
    fuel_load: float  # kg/m^2
    moisture_content: float  # 0-1
    height: float  # meters

@dataclass
class ClimateConditions:
    temperature: float  # Kelvin
    humidity: float  # 0-1
    precipitation: float  # mm/year

VEGETATION_PROPERTIES = {
    VegetationType.WATER: VegetationProperties(0.0, 1.0, 0.0),
    VegetationType.BARE_SOIL: VegetationProperties(0.1, 0.01, 0.0),
    VegetationType.GRASS: VegetationProperties(10.8, 0.03, 0.3),
    VegetationType.SHRUB: VegetationProperties(12.5, 0.04, 0.5),
    VegetationType.DECIDUOUS_TREE: VegetationProperties(16.0, 0.08, 6.0),
    VegetationType.CONIFEROUS_TREE: VegetationProperties(18.0, 0.08, 6.0),
    VegetationType.DEAD_VEGETATION: VegetationProperties(14.0, 0.01, 0.5),
}

BIOME_VEGETATION_PREFERENCES = {
    BiomeType.DESERT: {
        VegetationType.BARE_SOIL: 0.6,
        VegetationType.SHRUB: 0.3,
        VegetationType.GRASS: 0.1,
        VegetationType.DEAD_VEGETATION: 0.1
    },
    BiomeType.GRASSLAND: {
        VegetationType.GRASS: 0.7,
        VegetationType.SHRUB: 0.2,
        VegetationType.DECIDUOUS_TREE: 0.1,
        VegetationType.BARE_SOIL: 0.1
    },
    BiomeType.TEMPERATE_FOREST: {
        VegetationType.DECIDUOUS_TREE: 0.6,
        VegetationType.CONIFEROUS_TREE: 0.2,
        VegetationType.SHRUB: 0.15,
        VegetationType.GRASS: 0.1
    },
    BiomeType.BOREAL_FOREST: {
        VegetationType.CONIFEROUS_TREE: 0.7,
        VegetationType.SHRUB: 0.2,
        VegetationType.GRASS: 0.1,
        VegetationType.DEAD_VEGETATION: 0.1
    },
    BiomeType.ALPINE: {
        VegetationType.SHRUB: 0.4,
        VegetationType.GRASS: 0.3,
        VegetationType.BARE_SOIL: 0.2,
        VegetationType.CONIFEROUS_TREE: 0.1
    },
    BiomeType.WETLAND: {
        VegetationType.WATER: 0.3,
        VegetationType.GRASS: 0.4,
        VegetationType.SHRUB: 0.2,
        VegetationType.DECIDUOUS_TREE: 0.1
    }
}

# A pre-defined set of 2D unit gradient vectors for fast lookup
# These are the 8 vectors pointing to the midpoints and corners of a square.
# Normalizing them ensures the output range is consistent.
_GRADIENT_VECTORS = np.array([
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, 1], [1, -1], [-1, -1]
]) / np.sqrt(2)

# A global permutation table that will be initialized once per rng seed
_perlin_permutation_table = None

def _initialize_permutation_table(rng: np.random.Generator):
    """
    Initializes a global permutation table using a given random number generator.
    This ensures that the table is deterministic based on the rng's seed.
    """
    global _perlin_permutation_table
    if _perlin_permutation_table is not None:
        return
        
    _perlin_permutation_table = np.arange(256, dtype=np.int32)
    rng.shuffle(_perlin_permutation_table)
    _perlin_permutation_table = np.concatenate([_perlin_permutation_table, _perlin_permutation_table])

def perlin_noise_2d(x: float, y: float, rng: np.random.Generator, grid_size: float = 1.0) -> float:
    """
    Generate 2D Perlin noise using a fast, deterministic approach.

    Args:
        x (float): The x-coordinate to evaluate.
        y (float): The y-coordinate to evaluate.
        rng (np.random.Generator): A NumPy random number generator for seeding.
        grid_size (float): The size of the grid cells for noise generation.

    Returns:
        float: The Perlin noise value in the range [-1, 1].
    """
    global _perlin_permutation_table
    _initialize_permutation_table(rng)
    p = _perlin_permutation_table

    # Grid cell coordinates
    x_int, y_int = int(x // grid_size), int(y // grid_size)
    x_frac, y_frac = x / grid_size - x_int, y / grid_size - y_int

    # A simple, fast hash function to get an index for the gradient vectors
    def hash_coords(cx: int, cy: int) -> int:
        return p[p[cx & 255] + cy & 255]

    # Get gradient vectors for the four corners using fast lookups
    g00 = _GRADIENT_VECTORS[hash_coords(x_int, y_int) % 8]
    g10 = _GRADIENT_VECTORS[hash_coords(x_int + 1, y_int) % 8]
    g01 = _GRADIENT_VECTORS[hash_coords(x_int, y_int + 1) % 8]
    g11 = _GRADIENT_VECTORS[hash_coords(x_int + 1, y_int + 1) % 8]

    # Calculate distance vectors from corners to the point
    d00 = np.array([x_frac, y_frac])
    d10 = np.array([x_frac - 1, y_frac])
    d01 = np.array([x_frac, y_frac - 1])
    d11 = np.array([x_frac - 1, y_frac - 1])

    # Compute dot products between gradients and distance vectors
    n00 = np.dot(g00, d00)
    n10 = np.dot(g10, d10)
    n01 = np.dot(g01, d01)
    n11 = np.dot(g11, d11)

    # Fade function (quintic curve) for smooth interpolation
    def fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Smooth interpolation weights
    wx, wy = fade(x_frac), fade(y_frac)

    # Interpolate along x, then y
    nx0 = n00 + wx * (n10 - n00)
    nx1 = n01 + wx * (n11 - n01)
    
    # Final interpolation along y
    noise_val = nx0 + wy * (nx1 - nx0)

    # The output is already normalized to roughly [-1, 1] due to unit gradients
    return noise_val

class TerrainGenerator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        
    def generate_multi_scale_terrain(
        self,
        nx: int, ny: int, dx: float, dy: float,
        max_elevation: float,
        base_frequency: float = 0.01,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        ridged_weight: float = 0.3,
        base_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Enhanced terrain generation with ridged noise for mountain ranges and normalization."""
        if base_map is not None:
            height_map = base_map.copy()
        else:
            height_map = np.zeros((nx, ny), dtype=np.float64)  # Use float64 for precision
            
        terrain_seeds = [self.rng.integers(0, 2**32 - 1) for _ in range(octaves)]
        
        # Multi-octave Perlin noise
        for octave in range(octaves):
            frequency = base_frequency * (lacunarity ** octave)
            amplitude = persistence ** octave
            octave_map = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    x = i * frequency * dx
                    y = j * frequency * dy
                    
                    # Use NumPy-based Perlin noise
                    seed_rng = np.random.default_rng(terrain_seeds[octave])
                    perlin_val = perlin_noise_2d(x, y, seed_rng)
                    
                    # Ridged noise for mountain features
                    ridged_val = 1.0 - abs(perlin_val)
                    ridged_val = ridged_val ** 2
                    
                    # Blend regular and ridged noise
                    noise_val = ((1 - ridged_weight) * perlin_val + 
                               ridged_weight * ridged_val)
                    
                    octave_map[i, j] = noise_val
            
            height_map += octave_map * max_elevation * amplitude
        
        # Normalize the height map to span 0 to max_elevation
        height_min, height_max = np.min(height_map), np.max(height_map)
        if height_max > height_min:  # Avoid division by zero
            height_map = (height_map - height_min) / (height_max - height_min) * max_elevation
        else:
            height_map = np.zeros_like(height_map)  # Fallback if flat
            
        # Apply domain warping for more natural features
        height_map = self._apply_domain_warping(height_map, dx, dy)
        
        return np.clip(height_map, 0, max_elevation)
    
    def _apply_domain_warping(self, height_map: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Apply domain warping to create more natural terrain features."""
        nx, ny = height_map.shape
        warp_strength = 0.1
        
        # Create displacement maps
        disp_x = np.zeros((nx, ny))
        disp_y = np.zeros((nx, ny))
        
        warp_seed = self.rng.integers(0, 2**32 - 1)
        warp_rng = np.random.default_rng(warp_seed)
        
        for i in range(nx):
            for j in range(ny):
                x = i * 0.02 * dx
                y = j * 0.02 * dy
                
                disp_x[i, j] = perlin_noise_2d(x, y, warp_rng) * warp_strength
                disp_y[i, j] = perlin_noise_2d(x, y, np.random.default_rng(warp_seed + 1000)) * warp_strength
        
        # Apply displacement
        warped_map = np.zeros_like(height_map)
        for i in range(nx):
            for j in range(ny):
                new_i = int(np.clip(i + disp_x[i, j] * nx, 0, nx - 1))
                new_j = int(np.clip(j + disp_y[i, j] * ny, 0, ny - 1))
                warped_map[i, j] = height_map[new_i, new_j]
        
        return warped_map
    
    def generate_advanced_rivers(
        self,
        height_map: np.ndarray,
        min_flow_accumulation: int = 50,
        river_depth: float = 2.0,
        lake_threshold: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rivers using flow accumulation and drainage networks."""
        nx, ny = height_map.shape
        flow_dir = self._calculate_flow_direction(height_map)
        flow_acc = self._calculate_flow_accumulation(flow_dir)
        
        river_mask = flow_acc >= min_flow_accumulation
        modified_height = height_map.copy()
        
        for i in range(nx):
            for j in range(ny):
                if river_mask[i, j]:
                    depth = min(river_depth, river_depth * np.log(flow_acc[i, j] / min_flow_accumulation + 1))
                    depth = min(depth, modified_height[i, j])
                    modified_height[i, j] -= depth
        
        modified_height = np.clip(modified_height, 0, None)
        
        lake_mask = self._identify_lakes(modified_height, flow_acc, lake_threshold)
        water_level = np.where(lake_mask, modified_height, np.inf)
        
        for i in range(nx):
            for j in range(ny):
                if lake_mask[i, j]:
                    local_water_level = np.min(water_level[max(0, i-5):i+6, max(0, j-5):j+6])
                    if not np.isinf(local_water_level):
                        modified_height[i, j] = local_water_level
        
        modified_height = np.clip(modified_height, 0, None)
        
        return modified_height, river_mask | lake_mask
    
    def _calculate_flow_direction(self, height_map: np.ndarray) -> np.ndarray:
        """Calculate D8 flow direction."""
        nx, ny = height_map.shape
        flow_dir = np.zeros((nx, ny), dtype=np.int32)
        
        directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(nx):
            for j in range(ny):
                steepest_slope = 0
                steepest_dir = 0
                
                for d, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        slope = height_map[i, j] - height_map[ni, nj]
                        if slope > steepest_slope:
                            steepest_slope = slope
                            steepest_dir = d
                
                flow_dir[i, j] = steepest_dir
        
        return flow_dir
    
    def _calculate_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """Calculate flow accumulation using flow direction."""
        nx, ny = flow_dir.shape
        flow_acc = np.ones((nx, ny), dtype=np.int32)
        
        directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        
        for _ in range(10):
            for i in range(nx):
                for j in range(ny):
                    direction = flow_dir[i, j]
                    di, dj = directions[direction]
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < nx and 0 <= nj < ny:
                        flow_acc[ni, nj] += 1
        
        return flow_acc
    
    def _identify_lakes(self, height_map: np.ndarray, flow_acc: np.ndarray, threshold: float) -> np.ndarray:
        """Identify potential lake locations."""
        nx, ny = height_map.shape
        lake_mask = np.zeros((nx, ny), dtype=bool)
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if flow_acc[i, j] > 100:
                    neighbors = height_map[i-1:i+2, j-1:j+2]
                    if height_map[i, j] <= np.min(neighbors):
                        lake_mask[i, j] = True
        
        return lake_mask

class BiomeClassifier:
    def __init__(self):
        pass
    
    def classify_biomes(
        self,
        height_map: np.ndarray,
        climate: ClimateConditions,
        latitude: float = 45.0
    ) -> np.ndarray:
        """Classify biomes based on elevation, climate, and latitude."""
        nx, ny = height_map.shape
        biome_map = np.zeros((nx, ny), dtype=int)
        
        temp_lapse_rate = -6.5 / 1000.0
        
        for i in range(nx):
            for j in range(ny):
                elevation = height_map[i, j]
                local_temp = climate.temperature + elevation * temp_lapse_rate
                
                if elevation < 10:
                    biome_map[i, j] = BiomeType.WETLAND.value
                elif local_temp < 268:
                    biome_map[i, j] = BiomeType.ALPINE.value
                elif local_temp < 278:
                    biome_map[i, j] = BiomeType.BOREAL_FOREST.value
                elif climate.precipitation < 300:
                    biome_map[i, j] = BiomeType.DESERT.value
                elif climate.precipitation < 800:
                    biome_map[i, j] = BiomeType.GRASSLAND.value
                else:
                    biome_map[i, j] = BiomeType.TEMPERATE_FOREST.value
        
        return biome_map

class VegetationGenerator:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng
    
    def generate_realistic_vegetation(
        self,
        height_map: np.ndarray,
        water_mask: np.ndarray,
        biome_map: np.ndarray,
        dx: float, dy: float,
        climate: ClimateConditions,
        disturbance_probability: float = 0.05,
        succession_enabled: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate realistic vegetation with ecological rules."""
        nx, ny = height_map.shape
        veg_map = np.zeros((nx, ny), dtype=int)
        moisture_map = self._calculate_moisture_map(height_map, water_mask, climate)
        
        for i in range(nx):
            for j in range(ny):
                if water_mask[i, j]:
                    veg_map[i, j] = VegetationType.WATER.value
                else:
                    biome = BiomeType(biome_map[i, j])
                    veg_map[i, j] = self._select_vegetation_for_biome(biome, i, j, dx, dy)
        
        if succession_enabled:
            veg_map = self._apply_succession(veg_map, height_map, moisture_map)
        
        veg_map = self._apply_disturbances(veg_map, disturbance_probability)
        veg_map = self._apply_spatial_clustering(veg_map)
        
        fuel_map = self._calculate_fuel_loads(veg_map, climate)
        
        return veg_map, fuel_map, moisture_map
    
    def _calculate_moisture_map(
        self,
        height_map: np.ndarray,
        water_mask: np.ndarray,
        climate: ClimateConditions
    ) -> np.ndarray:
        """Calculate soil moisture based on elevation, water proximity, and climate."""
        nx, ny = height_map.shape
        moisture_map = np.full((nx, ny), climate.humidity)
        
        if np.any(water_mask):
            water_coords = np.column_stack(np.where(water_mask))
            if len(water_coords) > 0:
                for i in range(nx):
                    for j in range(ny):
                        if not water_mask[i, j]:
                            distances = np.sqrt(np.sum((water_coords - [i, j])**2, axis=1))
                            min_dist = np.min(distances)
                            water_influence = np.exp(-min_dist / 20.0) * 0.5
                            moisture_map[i, j] = min(1.0, moisture_map[i, j] + water_influence)
        
        elevation_factor = 1.0 - (height_map / np.max(height_map)) * 0.3
        moisture_map *= elevation_factor
        
        return np.clip(moisture_map, 0.1, 1.0)
    
    def _select_vegetation_for_biome(
        self,
        biome: BiomeType,
        i: int, j: int,
        dx: float, dy: float
    ) -> int:
        """Select vegetation type based on biome and local conditions."""
        preferences = BIOME_VEGETATION_PREFERENCES[biome]
        
        noise_seed = self.rng.integers(0, 2**32 - 1)
        noise_rng = np.random.default_rng(noise_seed)
        noise_val = perlin_noise_2d(i * 0.1 * dx, j * 0.1 * dy, noise_rng)
        noise_val = (noise_val + 1) / 2.0
        
        veg_types = list(preferences.keys())
        weights = [preferences[veg_type] * (1 + noise_val * 0.5) for veg_type in veg_types]
        
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            selected_veg = self.rng.choice(veg_types, p=weights)
            return selected_veg.value
        
        return VegetationType.BARE_SOIL.value
    
    def _apply_succession(
        self,
        veg_map: np.ndarray,
        height_map: np.ndarray,
        moisture_map: np.ndarray
    ) -> np.ndarray:
        """Apply ecological succession rules."""
        nx, ny = veg_map.shape
        succession_map = veg_map.copy()
        
        succession_rules = {
            VegetationType.BARE_SOIL: {VegetationType.GRASS: 0.3},
            VegetationType.GRASS: {VegetationType.SHRUB: 0.2, VegetationType.DECIDUOUS_TREE: 0.1},
            VegetationType.SHRUB: {VegetationType.DECIDUOUS_TREE: 0.15, VegetationType.CONIFEROUS_TREE: 0.1},
            VegetationType.DEAD_VEGETATION: {VegetationType.GRASS: 0.4, VegetationType.SHRUB: 0.2}
        }
        
        for i in range(nx):
            for j in range(ny):
                current_veg = VegetationType(veg_map[i, j])
                if current_veg in succession_rules:
                    for next_veg, prob in succession_rules[current_veg].items():
                        moisture_factor = moisture_map[i, j]
                        elevation_factor = 1.0 - (height_map[i, j] / np.max(height_map))
                        adjusted_prob = prob * moisture_factor * elevation_factor
                        if self.rng.random() < adjusted_prob:
                            succession_map[i, j] = next_veg.value
                            break
        
        return succession_map
    
    def _apply_disturbances(
        self,
        veg_map: np.ndarray,
        disturbance_probability: float
    ) -> np.ndarray:
        """Apply natural disturbances (fire, disease, etc.)."""
        nx, ny = veg_map.shape
        disturbed_map = veg_map.copy()
        
        disturbance_mask = self.rng.random((nx, ny)) < disturbance_probability
        
        for i in range(nx):
            for j in range(ny):
                if disturbance_mask[i, j]:
                    current_veg = VegetationType(veg_map[i, j])
                    if current_veg in [VegetationType.GRASS, VegetationType.SHRUB, 
                                     VegetationType.DECIDUOUS_TREE, VegetationType.CONIFEROUS_TREE]:
                        if self.rng.random() < 0.7:
                            disturbed_map[i, j] = VegetationType.DEAD_VEGETATION.value
                        else:
                            disturbed_map[i, j] = VegetationType.BARE_SOIL.value
        
        return disturbed_map
    
    def _apply_spatial_clustering(self, veg_map: np.ndarray) -> np.ndarray:
        """Apply spatial clustering to create more realistic vegetation patches."""
        nx, ny = veg_map.shape
        clustered_map = veg_map.copy()
        
        for _ in range(3):
            temp_map = clustered_map.copy()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    neighborhood = clustered_map[i-1:i+2, j-1:j+2]
                    unique, counts = np.unique(neighborhood, return_counts=True)
                    current_veg = clustered_map[i, j]
                    current_count = counts[unique == current_veg][0] if current_veg in unique else 0
                    if current_count < 3:
                        most_common_idx = np.argmax(counts)
                        most_common_veg = unique[most_common_idx]
                        if counts[most_common_idx] >= 5 and self._is_valid_transition(
                            VegetationType(current_veg), VegetationType(most_common_veg)
                        ):
                            temp_map[i, j] = most_common_veg
            clustered_map = temp_map
        
        return clustered_map
    
    def _is_valid_transition(self, from_veg: VegetationType, to_veg: VegetationType) -> bool:
        """Check if vegetation transition is ecologically valid."""
        valid_transitions = {
            VegetationType.BARE_SOIL: [VegetationType.GRASS, VegetationType.SHRUB],
            VegetationType.GRASS: [VegetationType.SHRUB, VegetationType.DECIDUOUS_TREE, VegetationType.BARE_SOIL],
            VegetationType.SHRUB: [VegetationType.DECIDUOUS_TREE, VegetationType.CONIFEROUS_TREE, VegetationType.GRASS],
            VegetationType.DECIDUOUS_TREE: [VegetationType.CONIFEROUS_TREE, VegetationType.SHRUB],
            VegetationType.CONIFEROUS_TREE: [VegetationType.DECIDUOUS_TREE, VegetationType.SHRUB],
            VegetationType.DEAD_VEGETATION: [VegetationType.GRASS, VegetationType.SHRUB, VegetationType.BARE_SOIL]
        }
        return to_veg in valid_transitions.get(from_veg, [])
    
    def _calculate_fuel_loads(
        self,
        veg_map: np.ndarray,
        climate: ClimateConditions
    ) -> np.ndarray:
        """Calculate fuel loads with environmental modifiers."""
        nx, ny = veg_map.shape
        fuel_map = np.zeros((nx, ny), dtype=float)
        
        for i in range(nx):
            for j in range(ny):
                veg_type = VegetationType(veg_map[i, j])
                base_fuel = VEGETATION_PROPERTIES[veg_type].fuel_load
                climate_factor = 1.0 + (climate.temperature - 288.15) / 30.0
                variation = self.rng.uniform(0.8, 1.2)
                fuel_map[i, j] = base_fuel * climate_factor * variation
        
        return np.clip(fuel_map, 0.0, 15.0)
    
    @staticmethod
    def _calculate_fuel_moisture(
        veg_map: np.ndarray,
        moisture_map: np.ndarray,
        climate: ClimateConditions
    ) -> np.ndarray:
        """Calculate fuel moisture content."""
        nx, ny = veg_map.shape
        fuel_moisture_map = np.zeros((nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                veg_type = VegetationType(veg_map[i, j])
                base_moisture = VEGETATION_PROPERTIES[veg_type].moisture_content
                soil_moisture_effect = moisture_map[i, j] * QuicFireConstants.ENV_MOISTURE_MODIFIER
                humidity_effect = climate.humidity * QuicFireConstants.ENV_MOISTURE_MODIFIER
                fuel_moisture_map[i, j] = base_moisture + soil_moisture_effect + humidity_effect
        
        return np.clip(fuel_moisture_map, QuicFireConstants.ENV_MIN_MOISTURE, QuicFireConstants.ENV_MAX_MOISTURE)

class SatelliteImageProcessor:
    """Placeholder for satellite image processing capabilities."""
    def __init__(self):
        pass
    
    def process_satellite_image(self, image_path: str) -> Dict:
        """Process satellite image to extract terrain and vegetation data."""
        return {
            'height_map': None,
            'vegetation_map': None,
            'water_mask': None,
            'confidence_map': None
        }
    
    def analyze_patterns(self, processed_data: Dict) -> Dict:
        """Analyze patterns in satellite data for procedural generation."""
        return {
            'terrain_parameters': {},
            'vegetation_parameters': {},
            'biome_distribution': {}
        }

# class TerrainVisualizer:
#     @staticmethod
#     def visualize_comprehensive_terrain(
#         height_map: np.ndarray,
#         veg_map: np.ndarray,
#         water_mask: np.ndarray,
#         biome_map: np.ndarray,
#         fuel_map: np.ndarray,
#         dx: float, dy: float,
#         title: str = "Comprehensive Terrain Analysis"
#     ):
#         """Create a comprehensive visualization of the generated terrain."""
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         nx, ny = height_map.shape
#         extent = [0, nx * dx, 0, ny * dy]
        
#         im1 = axes[0, 0].imshow(height_map, cmap='terrain', extent=extent, origin='lower')
#         axes[0, 0].set_title('Elevation')
#         axes[0, 0].set_xlabel('X (m)')
#         axes[0, 0].set_ylabel('Y (m)')
#         plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
        
#         veg_colors = ['blue', 'brown', 'lightgreen', 'olive', 'green', 'darkgreen', 'gray']
#         veg_cmap = ListedColormap(veg_colors)
#         im2 = axes[0, 1].imshow(veg_map, cmap=veg_cmap, extent=extent, origin='lower')
#         axes[0, 1].set_title('Vegetation')
#         axes[0, 1].set_xlabel('X (m)')
#         cbar2 = plt.colorbar(im2, ax=axes[0, 1], ticks=range(len(VegetationType)))
#         cbar2.ax.set_yticklabels([veg.name for veg in VegetationType])
        
#         axes[0, 2].imshow(height_map, cmap='terrain', extent=extent, origin='lower', alpha=0.7)
#         axes[0, 2].imshow(water_mask, cmap='Blues', extent=extent, origin='lower', alpha=0.8)
#         axes[0, 2].set_title('Water Bodies')
#         axes[0, 2].set_xlabel('X (m)')
        
#         biome_colors = ['yellow', 'lightgreen', 'green', 'darkgreen', 'white', 'lightblue']
#         biome_cmap = ListedColormap(biome_colors)
#         im4 = axes[1, 0].imshow(biome_map, cmap=biome_cmap, extent=extent, origin='lower')
#         axes[1, 0].set_title('Biomes')
#         axes[1, 0].set_xlabel('X (m)')
#         axes[1, 0].set_ylabel('Y (m)')
#         cbar4 = plt.colorbar(im4, ax=axes[1, 0], ticks=range(len(BiomeType)))
#         cbar4.ax.set_yticklabels([biome.name for biome in BiomeType])
        
#         im5 = axes[1, 1].imshow(fuel_map, cmap='YlOrRd', extent=extent, origin='lower')
#         axes[1, 1].set_title('Fuel Load')
#         axes[1, 1].set_xlabel('X (m)')
#         plt.colorbar(im5, ax=axes[1, 1], label='Fuel Load (kg/m²)')
        
#         plt.tight_layout()
#         plt.suptitle(title, fontsize=16, y=0.98)
#         plt.show()
    
#     @staticmethod
#     def create_3d_terrain_with_vegetation(
#         height_map: np.ndarray,
#         veg_map: np.ndarray,
#         dx: float, dy: float,
#         title: str = "3D Terrain with Vegetation"
#     ):
#         """Create enhanced 3D visualization with vegetation coloring."""
#         nx, ny = height_map.shape
#         fig = plt.figure(figsize=(14, 10))
#         ax = fig.add_subplot(111, projection='3d')
        
#         X = np.arange(nx) * dx
#         Y = np.arange(ny) * dy
#         X, Y = np.meshgrid(X, Y, indexing='ij')
        
#         veg_colors = {
#             VegetationType.WATER.value: [0.2, 0.4, 0.8, 1.0],
#             VegetationType.BARE_SOIL.value: [0.6, 0.4, 0.2, 1.0],
#             VegetationType.GRASS.value: [0.4, 0.8, 0.2, 1.0],
#             VegetationType.SHRUB.value: [0.6, 0.6, 0.0, 1.0],
#             VegetationType.DECIDUOUS_TREE.value: [0.2, 0.6, 0.2, 1.0],
#             VegetationType.CONIFEROUS_TREE.value: [0.1, 0.4, 0.1, 1.0],
#             VegetationType.DEAD_VEGETATION.value: [0.4, 0.4, 0.4, 1.0]
#         }
        
#         colors = np.zeros((nx, ny, 4))
#         for i in range(nx):
#             for j in range(ny):
#                 veg_type = veg_map[i, j]
#                 colors[i, j] = veg_colors.get(veg_type, [0.5, 0.5, 0.5, 1.0])
        
#         surf = ax.plot_surface(X, Y, height_map, 
#                               facecolors=colors, 
#                               linewidth=0, 
#                               antialiased=True,
#                               alpha=0.9)
        
#         ax.set_xlabel('X (m)')
#         ax.set_ylabel('Y (m)')
#         ax.set_zlabel('Elevation (m)')
#         ax.set_title(title)
        
#         max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), height_map.max()-height_map.min()]).max() / 2.0
#         mid_x = (X.max()+X.min()) * 0.5
#         mid_y = (Y.max()+Y.min()) * 0.5
#         mid_z = (height_map.max()+height_map.min()) * 0.5
#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
#         ax.view_init(elev=25, azim=-60)
        
#         plt.tight_layout()
#         plt.show()

def generate_realistic_environment(
    nx: int = 100,
    ny: int = 100,
    dx: float = 10.0,
    dy: float = 10.0,
    max_elevation: float = 500.0,
    base_frequency: float = 0.005,
    ridged_weight: float = 0.005,
    min_flow_accumulation: int = 50,
    river_depth: float = 2.0,
    lake_threshold: float = 0.1,
    climate: Dict = None,
    terrain_rng: Optional[np.random.Generator] = None,
    vegetation_rng: Optional[np.random.Generator] = None
) -> Dict:
    """Generate a complete realistic environment for fire simulation."""
    terrain_gen = TerrainGenerator(terrain_rng)
    biome_classifier = BiomeClassifier()
    veg_gen = VegetationGenerator(vegetation_rng)
    
    print("Generating terrain...")
    height_map = terrain_gen.generate_multi_scale_terrain(
        nx, ny, dx, dy, max_elevation,
        base_frequency=base_frequency,
        ridged_weight=ridged_weight
    )
    
    height_map, water_mask = terrain_gen.generate_advanced_rivers(
        height_map, min_flow_accumulation, river_depth, lake_threshold
    )
    
    print("Classifying biomes...")
    biome_map = biome_classifier.classify_biomes(height_map, climate)
    
    print("Generating vegetation...")
    veg_map, fuel_map, moisture_map = veg_gen.generate_realistic_vegetation(
        height_map, water_mask, biome_map, dx, dy, climate
    )

    fuel_moisture_map = VegetationGenerator._calculate_fuel_moisture(
        veg_map, moisture_map, climate
    )
    
    environment = {
        'height_map': height_map,
        'vegetation_map': veg_map,
        'fuel_map': fuel_map,
        'moisture_map': moisture_map,
        'water_mask': water_mask,
        'biome_map': biome_map,
        'climate': climate,
        'fuel_moisture_map': fuel_moisture_map,
        'grid_info': {
            'nx': nx, 'ny': ny, 'dx': dx, 'dy': dy,
            'max_elevation': max_elevation
        }
    }
    
    return environment

# def generate_and_save_terrain_series(
#     num_maps: int,
#     seeds: List[int],
#     nx: int = 100,
#     ny: int = 100,
#     dx: float = 2.0,
#     dy: float = 2.0,
#     max_elevation: float = 50.0,
#     base_frequency: float = 0.005,
#     ridged_weight: float = 0.005,
#     min_flow_accumulation: int = 50,
#     river_depth: float = 2.0,
#     lake_threshold: float = 0.1,
#     climate: Optional[Dict] = None,
#     terrain_output_dir: str = "terrain_maps",
#     vegetation_output_dir: str = "vegetation_maps"
# ) -> None:
#     """
#     Generate a series of terrain and vegetation maps and save them as separate 2D images.
    
#     Args:
#         num_maps: Number of terrain/vegetation map pairs to generate
#         seeds: List of random seeds for terrain and vegetation generation
#         nx, ny: Grid dimensions
#         dx, dy: Cell size in meters
#         max_elevation: Maximum terrain elevation in meters
#         base_frequency: Base frequency for terrain generation
#         ridged_weight: Weight for ridged noise in terrain generation
#         min_flow_accumulation: Minimum flow accumulation for rivers
#         river_depth: Depth of river carving
#         lake_threshold: Threshold for lake formation
#         climate: Climate conditions dictionary
#         terrain_output_dir: Directory to save terrain images
#         vegetation_output_dir: Directory to save vegetation images
#     """
#     # Create output directories if they don't exist
#     os.makedirs(terrain_output_dir, exist_ok=True)
#     os.makedirs(vegetation_output_dir, exist_ok=True)

#     global _perlin_permutation_table
#     for i, seed in enumerate(seeds[:num_maps], 1):
#         print(f"Generating map {i} with seed {seed}...")
        
#         # Reset the global permutation table to ensure unique terrain for each seed
#         _perlin_permutation_table = None
        
#         # Initialize random number generators
#         coll = RNGCollection(seed)
#         terrain_rng = coll.get_rng("terrain")
#         vegetation_rng = coll.get_rng("vegetation")

#         # Generate environment
#         environment = generate_realistic_environment(
#             nx=nx,
#             ny=ny,
#             dx=dx,
#             dy=dy,
#             max_elevation=max_elevation,
#             base_frequency=base_frequency,
#             ridged_weight=ridged_weight,
#             min_flow_accumulation=min_flow_accumulation,
#             river_depth=river_depth,
#             lake_threshold=lake_threshold,
#             climate=climate,
#             terrain_rng=terrain_rng,
#             vegetation_rng=vegetation_rng
#         )

#         # Extract relevant maps
#         height_map = environment['height_map']
#         veg_map = environment['vegetation_map']
#         extent = [0, nx * dx, 0, ny * dy]

#         # Create terrain plot
#         plt.figure(figsize=(6, 6))
#         im = plt.imshow(height_map, cmap='terrain', extent=extent, origin='lower')
#         plt.title(f'Terrain Elevation {i}')
#         plt.xlabel('X (m)')
#         plt.ylabel('Y (m)')
#         plt.colorbar(im, label='Elevation (m)')
#         plt.tight_layout()
        
#         # Save terrain plot
#         terrain_filename = os.path.join(terrain_output_dir, f'terrain_{i}.png')
#         plt.savefig(terrain_filename, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"Saved terrain map: {terrain_filename}")

#         # Create vegetation plot
#         plt.figure(figsize=(6, 6))
#         veg_colors = ['blue', 'brown', 'lightgreen', 'olive', 'green', 'darkgreen', 'gray']
#         veg_cmap = ListedColormap(veg_colors)
#         im = plt.imshow(veg_map, cmap=veg_cmap, extent=extent, origin='lower')
#         plt.title(f'Vegetation Map {i}')
#         plt.xlabel('X (m)')
#         plt.ylabel('Y (m)')
#         cbar = plt.colorbar(im, ticks=range(len(VegetationType)))
#         cbar.ax.set_yticklabels([veg.name for veg in VegetationType])
#         plt.tight_layout()
        
#         # Save vegetation plot
#         veg_filename = os.path.join(vegetation_output_dir, f'vegetation_{i}.png')
#         plt.savefig(veg_filename, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"Saved vegetation map: {veg_filename}")

# def create_gifs_from_images(
#     terrain_input_dir: str = "terrain_maps",
#     vegetation_input_dir: str = "vegetation_maps",
#     terrain_gif_output: str = "terrain_animation.gif",
#     vegetation_gif_output: str = "vegetation_animation.gif",
#     duration: int = 500,
#     loop: int = 0
# ) -> None:
#     """
#     Create animated GIFs from terrain and vegetation images.

#     Args:
#         terrain_input_dir: Directory containing terrain images (terrain_1.png, terrain_2.png, ...)
#         vegetation_input_dir: Directory containing vegetation images (vegetation_1.png, vegetation_2.png, ...)
#         terrain_gif_output: Output filename for terrain GIF
#         vegetation_gif_output: Output filename for vegetation GIF
#         duration: Duration of each frame in milliseconds
#         loop: Number of loops (0 for infinite)
#     """
#     # Helper function to load images in numerical order
#     def load_images(directory: str, prefix: str) -> List[Image.Image]:
#         images = []
#         i = 1
#         while True:
#             filename = os.path.join(directory, f"{prefix}_{i}.png")
#             if not os.path.exists(filename):
#                 break
#             img = Image.open(filename).convert('RGB')
#             images.append(img)
#             i += 1
#         return images

#     # Load terrain images
#     terrain_images = load_images(terrain_input_dir, "terrain")
#     if terrain_images:
#         print(f"Creating terrain GIF with {len(terrain_images)} frames...")
#         terrain_images[0].save(
#             terrain_gif_output,
#             save_all=True,
#             append_images=terrain_images[1:],
#             duration=duration,
#             loop=loop
#         )
#         print(f"Saved terrain GIF: {terrain_gif_output}")
#     else:
#         print(f"No terrain images found in {terrain_input_dir}")

#     # Load vegetation images
#     vegetation_images = load_images(vegetation_input_dir, "vegetation")
#     if vegetation_images:
#         print(f"Creating vegetation GIF with {len(vegetation_images)} frames...")
#         vegetation_images[0].save(
#             vegetation_gif_output,
#             save_all=True,
#             append_images=vegetation_images[1:],
#             duration=duration,
#             loop=loop
#         )
#         print(f"Saved vegetation GIF: {vegetation_gif_output}")
#     else:
#         print(f"No vegetation images found in {vegetation_input_dir}")

# if __name__ == "__main__":
#     # Example usage
#     climate = ClimateConditions(
#         temperature=260,
#         humidity=0.6,
#         precipitation=600.0
#     )

#     num_maps = 15
    
#     # Generate 5 random seeds
#     base_seed = 13434
#     random_seeds = [base_seed + i * 1000 for i in range(num_maps)]
    
#     # Generate and save terrain/vegetation images
#     generate_and_save_terrain_series(
#         num_maps=num_maps,
#         seeds=random_seeds,
#         climate=climate,
#         terrain_output_dir="output/terrain_maps",
#         vegetation_output_dir="output/vegetation_maps"
#     )
    
#     # Create GIFs from the generated images
#     create_gifs_from_images(
#         terrain_input_dir="output/terrain_maps",
#         vegetation_input_dir="output/vegetation_maps",
#         terrain_gif_output="output/terrain_animation.gif",
#         vegetation_gif_output="output/vegetation_animation.gif",
#         duration=500,  # 500ms per frame
#         loop=0  # Loop indefinitely
#     )

if __name__ == "__main__":
    from RNGCollection import RNGCollection
    import hashlib

    master_seed = 123453453
    climate = ClimateConditions(
        temperature=260,
        humidity=0.6,
        precipitation=600.0
    )

    coll = RNGCollection(master_seed)
    terrain_rng = coll.get_rng("terrain")
    vegetation_rng = coll.get_rng("vegetation")

    print(f"Master seed: {master_seed}")
    print(f"Terrain seed: {coll.get_seed('terrain')}")
    print(f"Vegetation seed: {coll.get_seed('vegetation')}")

    environment = generate_realistic_environment(
        nx=100, ny=100, dx=2.0, dy=2.0, max_elevation=50.0,
        climate=climate, terrain_rng=terrain_rng, vegetation_rng=vegetation_rng
    )
    
    np.save("height_map.npy", environment['height_map'])
    np.save("vegetation_map.npy", environment['vegetation_map'])
    with open("height_map.npy", "rb") as f:
        print(f"Height map MD5: {hashlib.md5(f.read()).hexdigest()}")
    with open("vegetation_map.npy", "rb") as f:
        print(f"Vegetation map MD5: {hashlib.md5(f.read()).hexdigest()}")

    # print("Creating visualizations...")
    # TerrainVisualizer.visualize_comprehensive_terrain(
    #     environment['height_map'],
    #     environment['vegetation_map'],
    #     environment['water_mask'],
    #     environment['biome_map'],
    #     environment['fuel_map'],
    #     environment['grid_info']['dx'],
    #     environment['grid_info']['dy'],
    #     "Enhanced Terrain Generation Results"
    # )
    
    # TerrainVisualizer.create_3d_terrain_with_vegetation(
    #     environment['height_map'],
    #     environment['vegetation_map'],
    #     environment['grid_info']['dx'],
    #     environment['grid_info']['dy'],
    #     "3D Terrain with Realistic Vegetation"
    # )

    print("MIN: ", np.min(environment['height_map']))
    
    print("\n--- Environment Statistics ---")
    print(f"Terrain elevation range: {np.min(environment['height_map']):.1f}m to {np.max(environment['height_map']):.1f}m")
    print(f"Water coverage: {np.sum(environment['water_mask']) / environment['water_mask'].size * 100:.1f}%")
    
    unique_veg, veg_counts = np.unique(environment['vegetation_map'], return_counts=True)
    print("\nVegetation distribution:")
    for veg_val, count in zip(unique_veg, veg_counts):
        veg_type = VegetationType(veg_val)
        percentage = count / environment['vegetation_map'].size * 100
        print(f"  {veg_type.name}: {percentage:.1f}%")
    
    unique_biomes, biome_counts = np.unique(environment['biome_map'], return_counts=True)
    print("\nBiome distribution:")
    for biome_val, count in zip(unique_biomes, biome_counts):
        biome_type = BiomeType(biome_val)
        percentage = count / environment['biome_map'].size * 100
        print(f"  {biome_type.name}: {percentage:.1f}%")
    
    print(f"\nFuel load range: {np.min(environment['fuel_map']):.2f} to {np.max(environment['fuel_map']):.2f} kg/m²")
    print(f"Average fuel load: {np.mean(environment['fuel_map']):.2f} kg/m²")
    
    print("\n--- Fire Simulation Ready ---")
    print("The environment data is now ready for fire simulation integration.")
    print("Fire inputs include: slope, aspect, fuel moisture, wind effects, and ignition thresholds.")