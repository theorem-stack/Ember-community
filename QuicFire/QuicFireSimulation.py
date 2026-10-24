from typing import Dict, Optional, Tuple, List
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireEngine import QuicFireEngine
from QuicFire.RNGCollection import RNGCollection
from QuicFire.QuicFireEnvGenerator import ClimateConditions, generate_realistic_environment
import random
import traceback
import json
import numpy as np

### ---- QUIC-Fire Simulation Control ---- ###

def run_fire_simulation(
    base_config: Dict,
    module_seeds: RNGCollection,
    climate: ClimateConditions, # module_rngs needs to be passed correctly if used internally by QuicFireEngine
    verbose: bool = True,
    visualize: bool = True
) -> Dict:
    """
    Run a fire simulation with generated or custom inputs.
    
    Args:
        base_config: Base simulation configuration
        module_rngs: RNGCollection instance for random number generation.
        verbose: Print detailed setup information
        visualize: Show fire movement visualization
    
    Returns:
        Simulation results dictionary
    """

    # DEBUG: Where do you put this? Feed only necessary information to the Engine. Store the rest outside.
    # Generate a complete environment
    environment = generate_realistic_environment(
        nx=base_config['nx'], 
        ny=base_config['ny'], 
        dx=base_config['dx'], 
        dy=base_config['dy'], 
        max_elevation=QuicFireConstants.MAX_TERRAIN_ELEVATION, 
        climate=climate, 
        terrain_rng=module_seeds.get_rng('terrain'),
        vegetation_rng=module_seeds.get_rng('vegetation'),
    )

    height_map = environment['height_map']
    veg_map = environment['vegetation_map']
    fuel_map = environment['fuel_map']
    fuel_moisture_map = environment['fuel_moisture_map']

    # Initialize simulation
    sim = QuicFireEngine(
        config=base_config, 
        veg_map=veg_map, 
        fuel_map=fuel_map, 
        fuel_moisture_map=fuel_moisture_map,
        height_map=height_map, 
        module_seeds=module_seeds
    )

    print("MODULE SEEDS: ", sim.module_seeds.module_seeds)

    # MODULE SEEDS:  {'master': 42, 'terrain': 383329928, 'vegetation': 3324115917, 'wind': 2811363265, 'burn_box': 1884968545, 'ignition': 1859786276}
    # MODULE SEEDS:  {'master': 10224593, 'terrain': 984482331, 'vegetation': 162498447, 'wind': 370373556, 'burn_box': 292542091, 'ignition': 206001335}
    
    # Run simulation
    results = sim.run_simulation(
        save_interval=60, # Assuming you want to save every 60 seconds for visualization
        verbose=verbose,
    )
    
    # Add the consumption rate history to the results dictionary
    results['consumption_rate_history'] = sim.consumption_rate_history
    
    # Print results
    if verbose:
        print_simulation_results(results)
    
    # Visualize if requested
    if visualize:
        # Conditional import
        from QuicFire.QuicFireVisualizer import visualize_fire_movement, visualize_results
        
        visualize_results(
            height_map=sim.height_map, 
            fuel_density=sim.fuel_density, 
            nx=sim.nx, 
            ny=sim.ny, 
            nz=sim.nz, 
            dx=sim.dx, 
            dy=sim.dy, 
            z_levels=sim.z_levels
        )
        visualize_fire_movement(results, base_config['nx'], base_config['ny'], float(base_config.get('dx', 2.0)), float(base_config.get('dy', 2.0)), sim.burn_box)
    
    return results

def print_simulation_results(results: Dict):
    """Print formatted simulation results."""
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    # Check for 'fires' and 'embers' keys to avoid KeyError if not always present
    print(f"Active fires: {len(results.get('fires', []))}")
    print(f"Ember locations: {len(results.get('embers', []))}")
    print(f"Final diagnostics: {results.get('diagnostics', 'N/A')}")
    print(f"Plume parameters: {results.get('plume', 'N/A')}")
    if 'time_series' in results:
        print(f"Time series data points: {len(results['time_series'])}")
    if 'consumption_rate_history' in results:
        # Consider taking an average over a stable period, not just the last value
        avg_burn_rate = np.mean(results['consumption_rate_history']) if results['consumption_rate_history'] else 0.0
        print(f"Avg. Mass Burn Rate (kg/m²/s): {avg_burn_rate:.4f}")


# --- Calibration Function ---
def calibrate_simulation_params(
    base_config: Dict,
    params_to_calibrate: Dict[str, Tuple[float, float]], # {param_name: (min_val, max_val)}
    target_avg_mass_burn_rate: float, # e.g., 0.02 kg/m²/s
    num_calibration_runs: int = 100,
    calibration_method: str = "random_search", # or "grid_search" (simple for few params)
    verbose_calibration: bool = True,
    visualize_best_run: bool = True,
    output_calibration_file: Optional[str] = "calibration_results.json"
) -> Dict:
    """
    Iteratively adjusts specified QuicFireConstants parameters to achieve a target average mass burn rate.

    Args:
        base_config: Base simulation configuration (grid size, total_time, etc.).
                     This config should represent your target scenario (e.g., EAFB).
        params_to_calibrate: A dictionary mapping QuicFireConstants attribute names (strings)
                             to a tuple (min_value, max_value) for their search range.
                             Example: {'CONSUMPTION_RATE': (0.5, 2.0), 'C_IGSC': (0.01, 0.1)}
        target_avg_mass_burn_rate: The desired average mass burn rate (kg/m²/s).
        num_calibration_runs: The number of parameter combinations to test.
        calibration_method: 'random_search' (recommended) or 'grid_search' (for very few params).
        verbose_calibration: If True, print progress during calibration runs.
        visualize_best_run: If True, visualize the simulation with the best-found parameters.
        output_calibration_file: Path to save calibration results. Set to None to skip.

    Returns:
        A dictionary containing the best-found parameters and calibration history.
    """
    print(f"\n--- Starting Calibration for Target Avg. Mass Burn Rate: {target_avg_mass_burn_rate:.4f} kg/m²/s ---")
    print(f"Parameters to calibrate: {list(params_to_calibrate.keys())}")
    print(f"Number of runs: {num_calibration_runs}")

    best_loss = float('inf')
    best_params = {}
    calibration_history: List[Dict] = []

    # Get original QuicFireConstants values to restore them later
    original_constants = {attr: getattr(QuicFireConstants, attr) for attr in params_to_calibrate.keys()}
    climate = ClimateConditions(temperature=260, humidity=0.6, precipitation=600.0)

    # Helper to generate parameter combinations
    def _generate_param_combinations_grid():
        # For simplicity, implementing a basic grid search for 2 params for now.
        # For more params, you'd need itertools.product or recursive generation.
        if len(params_to_calibrate) > 2:
            print("Warning: Grid search for >2 parameters is computationally very expensive. Consider 'random_search'.")
        
        param_names = list(params_to_calibrate.keys())
        param_ranges = [np.linspace(v[0], v[1], 5) for v in params_to_calibrate.values()] # Reduced steps to 5 for quicker grid test
        
        if len(param_names) == 1:
            for val0 in param_ranges[0]:
                yield {param_names[0]: val0}
        elif len(param_names) == 2:
            for val0 in param_ranges[0]:
                for val1 in param_ranges[1]:
                    yield {param_names[0]: val0, param_names[1]: val1}
        else: # Fallback to random search for >2 params in grid_search
            print("Grid search beyond 2 parameters not fully implemented for simple example. Falling back to random search.")
            # **IMPORTANT:** If you actually intend to fall back, you'd need to call and return its generator
            yield from _generate_param_combinations_random() 

    def _generate_param_combinations_random():
        for _ in range(num_calibration_runs):
            param_set = {}
            for param_name, (min_val, max_val) in params_to_calibrate.items():
                param_set[param_name] = random.uniform(min_val, max_val)
            yield param_set

    # Call the appropriate generator function to get the iterable
    param_generator = None
    if calibration_method == "grid_search":
        param_generator = _generate_param_combinations_grid()
    else: # Default to random search
        param_generator = _generate_param_combinations_random()

    # The fix: Iterate over the *result* of the generator function call
    for i, current_param_set in enumerate(param_generator):
        current_run_seed = random.randint(1, 1_000_000) # Use a fresh seed for each calibration run

        # Apply current parameter set to QuicFireConstants
        for param_name, param_value in current_param_set.items():
            setattr(QuicFireConstants, param_name, param_value)
        
        # Merge base_config with specific run parameters (e.g., current_run_seed)
        run_config = {**base_config, 'random_seed': current_run_seed}
        
        current_avg_burn_rate = 0.0
        try:
            # Random Number Generator
            module_seeds = RNGCollection(base_config['random_seed'])

            # Re-initialize module_rngs for each run for seed consistency
            results = run_fire_simulation(
                base_config=run_config,
                module_seeds=module_seeds,
                climate=climate,
                verbose=False,
                visualize=False
            )
            
            # Calculate average burn rate from the stable period (e.g., after initial ramp-up)
            # You might need to adjust this depending on your fire's behavior.
            # For simplicity, let's take the mean of the entire history for now.
            if results['consumption_rate_history']:
                current_avg_burn_rate = np.mean(results['consumption_rate_history'])
            else:
                current_avg_burn_rate = 0.0 # No fire / no consumption data

            loss = abs(current_avg_burn_rate - target_avg_mass_burn_rate)

            if verbose_calibration:
                param_str = ", ".join([f"{k}={v:.4f}" for k, v in current_param_set.items()])
                print(f"Calib Run {i+1}/{num_calibration_runs}: Params=[{param_str}], Avg. Burn Rate={current_avg_burn_rate:.4f}, Loss={loss:.4f}")

            calibration_history.append({
                'run': i + 1,
                'params': current_param_set,
                'simulated_avg_burn_rate': current_avg_burn_rate,
                'loss': loss
            })

            if loss < best_loss:
                best_loss = loss
                best_params = current_param_set.copy() # Important: copy the dict

        except Exception as e:
            if verbose_calibration:
                param_str = ", ".join([f"{k}={v:.4f}" for k, v in current_param_set.items()])
                print(f"Calib Run {i+1}/{num_calibration_runs}: Params=[{param_str}] FAILED with error: {e}")
                # traceback.print_exc() # Uncomment for full traceback during calibration
            calibration_history.append({
                'run': i + 1,
                'params': current_param_set,
                'simulated_avg_burn_rate': None, # Indicate failure
                'loss': float('inf'), # Max loss for failed runs
                'error': str(e)
            })
        finally:
            # Restore original QuicFireConstants values for the next iteration
            for attr, val in original_constants.items():
                setattr(QuicFireConstants, attr, val)
    
    print("\n--- Calibration Finished ---")
    if best_params:
        print(f"Best found parameters (Loss: {best_loss:.4f}):")
        for k, v in best_params.items():
            print(f"  {k}: {v:.6f}")
        
        if visualize_best_run:
            print("\nVisualizing best calibration run...")
            # Apply best params for visualization
            for param_name, param_value in best_params.items():
                setattr(QuicFireConstants, param_name, param_value)
            
            # Use a fresh RNG for visualization for consistency
            viz_module_seeds = RNGCollection(random.randint(1, 1_000_000))

            try:
                run_fire_simulation(base_config, viz_module_seeds, climate, verbose=True, visualize=True)
            except Exception as e:
                print(f"Failed to visualize best run: {e}")
            finally:
                # Restore original constants after visualization
                for attr, val in original_constants.items():
                    setattr(QuicFireConstants, attr, val)

    else:
        print("No successful calibration runs found.")
    
    # Save calibration history
    if output_calibration_file:
        try:
            with open(output_calibration_file, 'w') as f:
                json.dump(calibration_history, f, indent=4)
            print(f"Calibration history saved to '{output_calibration_file}'")
        except IOError as e:
            print(f"Error saving calibration history to file: {e}")

    return {
        'best_params': best_params,
        'best_loss': best_loss,
        'calibration_history': calibration_history
    }


# --- Existing functions (copy-pasted from your prompt for completeness) ---

def stress_test_simulations(
    num_tests: int, 
    min_nx: int = 50, max_nx: int = 150,
    min_ny: int = 50, max_ny: int = 200,
    min_nz: int = 20, max_nz: int = 70,
    min_dx: int = 2, max_dx: int = 2,
    min_dy: int = 2, max_dy: int = 2,
    min_total_time: int = 3600, max_total_time: int = 14400,
    min_num_ignition_lines: int = 1, max_num_ignition_lines: int = 5,
    min_num_wind_updates: int = 1, max_num_wind_updates: int = 5,
    verbose: bool = False,
    visualize: bool = False,
    output_file: Optional[str] = "failed_simulations.json"
) -> List[Dict]:
    """
    Randomly generates and runs multiple simulations to stress test the engine.
    Records the random_seed and full configuration of any failed simulations.
    Optionally saves failed cases to a JSON file.
    """
    failed_simulations_data: List[Dict] = []

    print(f"\n--- Starting Stress Test: {num_tests} Simulations ---")

    for i in range(num_tests):
        current_seed = random.randint(1, 1_000_000)
        
        test_config = {
            'nx': random.randint(min_nx, max_nx),
            'ny': random.randint(min_ny, max_ny),
            'nz': random.randint(min_nz, max_nz),
            'dx': random.randint(min_dx, max_dx),
            'dy': random.randint(min_dy, max_dy),
            'random_seed': current_seed,
            'total_time': random.randint(min_total_time, max_total_time),
        }
        
        print(f"\n--- Running Test {i+1}/{num_tests} (Seed: {current_seed}) ---")
        if verbose:
            print(f"Test Configuration: {test_config}")

        try:
            # Need a new RNGCollection for each stress test run to ensure different seeds
            stress_module_seeds = RNGCollection(current_seed)

            climate = ClimateConditions(temperature=260, humidity=0.6, precipitation=600.0)
            results = run_fire_simulation(
                test_config,
                module_seeds=stress_module_seeds,
                climate=climate,
                verbose=verbose,
                visualize=visualize,
            )

            fuel_consumption = results.get('diagnostics', {}).get('consumption', {})
            total_fuel_consumed = (
                fuel_consumption.get('surface', 0) +
                fuel_consumption.get('midstory', 0) +
                fuel_consumption.get('canopy', 0)
            )

            if total_fuel_consumed <= 1e-6:
                raise ValueError("ERROR: Effectively no fuel consumed, suggesting fire did not propagate or ignite properly.")
            
            print(f"Test {i+1} (Seed: {current_seed}, Fuel Consumed: {total_fuel_consumed:.2f}): SUCCESS")
        except Exception as e:
            print(f"Test {i+1} (Seed: {current_seed}): FAILED with error: {e}")
            print("--- Full traceback ---")
            traceback.print_exc()
            print("----------------------")
            failed_simulations_data.append({
                'seed': current_seed,
                'params': test_config,
                'error': str(e)
            })
            
    print(f"\n--- Stress Test Finished ---")
    if failed_simulations_data:
        print(f"Failed Simulations: {len(failed_simulations_data)} failed out of {num_tests} tests.")
        print("Details of Failed Simulations:")
        for fail_data in failed_simulations_data:
            print(f"  Seed: {fail_data['seed']}, Parameters: {fail_data['params']}, Error: {fail_data['error']}")
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(failed_simulations_data, f, indent=4)
                print(f"Failed simulation data saved to '{output_file}'")
            except IOError as e:
                print(f"Error saving failed simulations to file: {e}")
    else:
        print(f"All {num_tests} simulations ran successfully! 🎉")
        
    return failed_simulations_data

def run_failed_simulation(
    failed_case_data: Dict,
    verbose: bool = True,
    visualize: bool = True
) -> Optional[Dict]:
    """
    Reruns a specific simulation using its recorded parameters.
    """
    seed = failed_case_data.get('seed')
    params = failed_case_data.get('params')

    if not params:
        print("Error: Invalid failed case data. 'params' key is missing.")
        return None

    print(f"\n--- Rerunning Failed Simulation (Seed: {seed}) ---")
    print(f"Parameters: {params}")

    try:
        # Need a new RNGCollection for this rerun
        rerun_module_rngs = RNGCollection(seed)

        climate = ClimateConditions(temperature=260, humidity=0.6, precipitation=600.0)
        results = run_fire_simulation(
            params,
            module_seeds=rerun_module_rngs,
            climate=climate,
            verbose=verbose,
            visualize=visualize,
        )
        print(f"Rerun of Seed {seed}: SUCCESS ✅")
        return results
    except Exception as e:
        print(f"Rerun of Seed {seed}: FAILED AGAIN with error: {e}")
        print("--- Full traceback ---")
        traceback.print_exc()
        print("----------------------")
        return None

if __name__ == "__main__":
    
    # Example 1: Generated scenario (your original automated case)
    print("=== GENERATED SCENARIO ===")
    # Base configuration for individual runs (your original automated case)
    base_config = {
        'nx': 100,
        'ny': 100,
        'nz': 50,
        'dx': 2,
        'dy': 2,
        'max_grid_height': QuicFireConstants.DEFAULT_MAX_GRID_HEIGHT,
        'random_seed': QuicFireConstants.DEFAULT_RANDOM_SEED,
        'total_time': 10000
    }

    # Define climate conditions
    climate = ClimateConditions(
        temperature=260,  # Kelvin
        humidity=0.6,
        precipitation=600.0  # mm/year
    )

    # Random Number Generator
    module_seeds = RNGCollection(base_config['random_seed'])

    result = run_fire_simulation(base_config=base_config, module_seeds=module_seeds, climate=climate, verbose=True)

    # Example 2: Stress Test
    # print("\n\n=== STRESS TEST SCENARIOS ===")
    # # Set output_file to None if you don't want to save to a file
    # failed_results_list = stress_test_simulations(
    #     num_tests=5, # Run 5 random simulations for quick test
    #     min_nx=50, max_nx=100,
    #     min_ny=50, max_ny=150,
    #     min_nz=30, max_nz=60,
    #     min_dx=1, max_dx=3,
    #     min_dy=1, max_dy=3,
    #     min_total_time=1800, max_total_time=7200,
    #     min_num_ignition_lines=1, max_num_ignition_lines=3,
    #     min_num_wind_updates=1, max_num_wind_updates=3,
    #     verbose=False,
    #     visualize=False,
    #     output_file="failed_simulations_log.json"
    # )

    # Example of loading from file and rerunning (if you ran the stress test previously and closed the script)
    # try:
    #     with open("failed_simulations_log.json", 'r') as f:
    #         loaded_failed_cases = json.load(f)
    #     if loaded_failed_cases:
    #         print("\n\n=== RERUNNING A FAILED SCENARIO FROM FILE ===")
    #         # Rerun the first loaded failed case
    #         run_failed_simulation(loaded_failed_cases[0], verbose=True, visualize=True) # Changed index to 0 for first case
    # except FileNotFoundError:
    #     print("\nNo 'failed_simulations_log.json' file found to load from.")

    # # --- Example 3: Calibration ---
    # print("\n\n=== CALIBRATION SCENARIO ===")
    # # IMPORTANT: The base_config for calibration should represent a typical scenario
    # # e.g., the EAFB specific fuel bed and domain size, not random sizes.
    # # Set total_time to be long enough for the fire to stabilize its burn rate.
    # calibration_base_config = {
    #     'nx': 50,
    #     'ny': 50,
    #     'nz': 50,
    #     'dx': 2,
    #     'dy': 2,
    #     'random_seed': 42, # Fixed seed for consistent calibration base
    #     'total_time': 3600 # Run for 1 hour to get stable burn rate
    # }

    # # Define parameters to calibrate and their search ranges
    # # These are examples; adjust min/max based on your knowledge and initial tests.
    # # Make sure these attributes exist in QuicFireConstants.
    # params_to_tune = {
    #     'CONSUMPTION_RATE': (0.01, 1.0),    # kg/m²/s
    #     'C_IGSC': (6.0, 8.0),               # Critical Ignition Surface Concentration
    #     'EMISSION_RATE': (0.1, 1.0),        # EPs emitted per heat release (adjust if fire spreads too fast/slow)
    #     'T_BURNOUT': (30, 180),             # Burn duration
    #     'DEFAULT_MOISTURE_INIT': (0.05, 0.15),      # Initial fuel moisture (QUIC-Fire paper)
    #     'DRYING_RATE': (0.005, 0.01),       # Moisture reduction rate
    #     'DT': (5.0, 20.0)                   # Time step (s)
    # }

    # target_burn_rate = 0.02 # Example target: 0.02 kg/m²/s (moderate fire)

    # calibration_results = calibrate_simulation_params(
    #     base_config=calibration_base_config,
    #     params_to_calibrate=params_to_tune,
    #     target_avg_mass_burn_rate=target_burn_rate,
    #     num_calibration_runs=50, # Start with a small number for testing
    #     calibration_method="random_search", # Recommended for more than 2 params
    #     verbose_calibration=True,
    #     visualize_best_run=True,
    #     output_calibration_file="calibration_log.json"
    # )

    # print("\nCalibration Process Complete.")
    # print(f"Best parameters found: {calibration_results.get('best_params', 'N/A')}")
    # print(f"Best loss achieved: {calibration_results.get('best_loss', 'N/A')}")