import numpy as np
from typing import Dict, Optional

class RNGCollection:
    """
    A collection of deterministic seeds for QuicFire modules.
    Generates module-specific numpy.random.Generator objects on-demand for reproducible random number generation.
    """
    def __init__(self, master_seed: int, initial_seeds: Optional[Dict[str, int]] = None):
        """
        Initializes the RNG collection with a master seed or provided module seeds.

        Args:
            master_seed (int): A master seed to derive module-specific seeds if not provided.
            initial_seeds (Optional[Dict[str, int]]): A dictionary of pre-initialized module seeds.
                                                     If provided, these override default generation.
        """
        self.master_seed = master_seed
        self.module_seeds: Dict[str, int] = {
            'master': master_seed,
            'terrain': None,
            'vegetation': None,
            'wind': None,
            'burn_box': None,
            'ignition': None
        }
        self._initialize_seeds(initial_seeds)

    def _initialize_seeds(self, initial_seeds: Optional[Dict[str, int]]):
        """Internal helper to set up module-specific seeds."""
        module_keys = ['terrain', 'vegetation', 'wind', 'burn_box', 'ignition']
        rng = np.random.default_rng(self.master_seed)

        if initial_seeds:
            for key in module_keys:
                if key in initial_seeds:
                    if not isinstance(initial_seeds[key], int):
                        raise TypeError(f"Provided seed for '{key}' is not an integer.")
                    self.module_seeds[key] = initial_seeds[key]
                else:
                    # Generate a deterministic seed by consuming RNG output
                    self.module_seeds[key] = rng.integers(0, 2**32)
        else:
            # Generate seeds in fixed order for all modules
            for key in module_keys:
                self.module_seeds[key] = rng.integers(0, 2**32)

    def get_rng(self, module: str) -> np.random.Generator:
        """
        Returns a new Generator with the deterministic seed for the specified module.

        Args:
            module (str): The module name ('master', 'terrain', 'vegetation', 'wind', 'burn_box', 'ignition').

        Returns:
            np.random.Generator: A Generator initialized with the module's seed.

        Raises:
            ValueError: If the module name is invalid.
        """
        if module not in self.module_seeds:
            raise ValueError(f"Invalid module name: {module}. Must be one of {list(self.module_seeds.keys())}")
        return np.random.default_rng(self.module_seeds[module])

    def replace_module_seed(self, module_key: str, new_seed: int) -> None:
        """
        Replaces the seed for a specific module.

        Args:
            module_key (str): The module name ('terrain', 'vegetation', 'wind', 'burn_box', 'ignition').
            new_seed (int): The new seed value for the module.

        Raises:
            ValueError: If the module_key is invalid.
            TypeError: If new_seed is not an integer.
        """
        if module_key not in self.module_seeds:
            raise ValueError(f"Invalid module name: {module_key}. Must be one of {list(self.module_seeds.keys())}")
        if not isinstance(new_seed, int):
            raise TypeError(f"New seed for '{module_key}' must be an integer.")
        self.module_seeds[module_key] = new_seed
    
    def get_seed(self, module_key: str) -> int:
        return self.module_seeds[module_key]