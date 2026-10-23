from QuicFire.QuicFireEnvGenerator import ClimateConditions
from QuicFire.QuicFireSimulation import run_fire_simulation
from QuicFire.RNGCollection import RNGCollection
from QuicFire.QuicFireConstants import QuicFireConstants

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
        'max_height': 200,
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