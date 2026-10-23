# Important Variables For Burn Rate
# DEFAULT_MOISTURE_INIT: Higher moisture rate slows burn rate. Check vegetation generator
# CONSUMPTION_RATE: Direct burn speed
# C_IGSC: Resistance to burn
# DT: Major effect on speed and burn rate

# MOISTURE_THRESHOLD = 0.2 # KEY
# HEAT_FLUX = 1000.0  # kW/m # KEY
# EMISSION_RATE = 0.01  # EPs per kW·s

class QuicFireConstants:
    """Constants for QuicFire simulation"""

    DEFUALT_SIMULATION_TIME = 10000 # seconds (s)
    DEFAULT_RANDOM_SEED = 12243261 # Dimensionless (12243261 THE ONE) (35234113 GREAT) (122461 REALLY GOOD)
    PROGRESS_INTERVAL = 60 # seconds (s)

    # Grid configuration
    DEFAULT_NX = 100 # Dimensionless (number of cells)
    DEFAULT_NY = 100 # Dimensionless (number of cells)
    DEFAULT_NZ = 50 # Dimensionless (number of levels)
    DEFAULT_DX = 2.0 # meters (m)
    DEFAULT_DY = 2.0 # meters (m)
    DEFAULT_MAX_GRID_HEIGHT = 300.0  # meters (m)
    
    SURFACE_HEIGHTS = {200: 150.0, 100: 100.0, 0: 50.0} # keys: Dimensionless (number of cells), values: meters (m)
    SURFACE_SPACINGS = {200: 1.0, 10: 2.0, 0: 0.5} # keys: Dimensionless (number of cells), values: meters (m)
    MIN_SURFACE_POINTS = 10 # Dimensionless (number of cells)

    # Physical constants
    GRAVITY = 9.81  # meters per second squared (m/s^2)
    T_AMBIENT = 300.0  # Kelvin (K)
    T_FIRE = 1000.0  # Kelvin (K)
    RHO_AIR = 1.225  # kilograms per cubic meter (kg/m^3)
    C_P = 1005.0  # Joules per kilogram-Kelvin (J/(kg·K)) - Specific Heat Capacity of Air
    H_SF = 5.0  # meters (m) - Surface Fuel Height (reference)
    HEAT_FLUX = 1000.0  # kilowatts per square meter (kW/m^2)
    KAPPA = 0.4 # Dimensionless - Von Karman constant
    Z0 = 0.1  # meters (m) - Roughness length
    U_REF = 5.0  # meters per second (m/s) - Reference wind speed
    Z_REF = 10.0  # meters (m) - Reference height for wind speed

    # Fire parameters
    C_IGSC = 6.0  # kilograms per cubic meter (kg/m^3) - Critical Ignition Surface Concentration (volumetric)
    A_V = 4000.0  # per meter (m^-1) - Empirical constant for convective heat transfer
    A_V_NORM = 4000.0  # per meter (m^-1) - Normalization constant for A_V
    LAMBDA_EP = 0.1 # Dimensionless - Ratio of thermal conductivity of EP to air (e.g., thermal diffusivity ratio)
    LAMBDA_AIR = 0.025 # Watts per meter-Kelvin (W/(m·K)) - Thermal conductivity of air

    RHO_F0 = 20.0  # kilograms per cubic meter (kg/m^3) - Reference/Maximum Fuel Density (Check Vegetation Properties)
    T_BURNOUT = 100.0  # seconds (s)
    EMISSION_RATE = 0.01  # Energy Packets per kilowatt-second (EPs/(kW·s))
    DT = 20.0  # seconds (s) - Simulation time step
    MAX_EP_COUNT = 50 # Dimensionless (number of EPs)
    BASE_ALTITUDE = 0 # meters (m)
    OXYGEN_ALTITUDE_DECAY_RATE = 0.000015 # m^-1
    EP_DECAY_RATE = 0.2 # Percentage of EPs removed from active EPs per time step (simplification)

    DEFAULT_MOISTURE_INIT = 0.08 # Dimensionless (fraction)
    ENV_MIN_MOISTURE = 0.01 # Percentage
    ENV_MAX_MOISTURE = 0.10 # Percentage
    ENV_MOISTURE_MODIFIER = 0.10 # Percentage of effect the climate humidity and ground moisture affect

    # Movement probabilities
    MOVEMENT_PROBS = {
        'stay': 0.25, # Dimensionless
        'east': 0.25, # Dimensionless
        'west': 0.25, # Dimensionless
        'north': 0.25, # Dimensionless
        'south': 0.25, # Dimensionless
        'up': 0.1, # Dimensionless
        'down': 0.05 # Dimensionless
    }
    TURBULENCE_FACTOR = 0.2 # Dimensionless
    RADIATIVE_FRACTION = 0.05 # Dimensionless (fraction)
    DRYING_RATE = 0.02 # Dimensionless (fraction of moisture per second or per DT, context dependent) - assuming per second to align with physics
    MOISTURE_THRESHOLD = 0.02 # Dimensionless (fraction)
    VERTICAL_TRANSPORT_PROB = 0.2 # Dimensionless (probability)
    VERTICAL_EP_TRANSFER = 0.2 # Dimensionless (fraction of EPs)

    # Fuel distribution
    FUEL_TEMPLATE_SOURCE = "example_fuel_template.png" # String (file path)
    TERRAIN_TEMPLATE_SOURCE = "example_terrain_template.png" # String (file path)
    FUEL_GRID_JSON_PATH = "fuel_grid.json" # String (file path)
    FUEL_HEIGHTS = {
        'litter': 0.5, # meters (m)
        'surface': 2.0, # meters (m)
        'midstory': 5.0, # meters (m)
        'canopy': 25.0 # meters (m)
    }
    FUEL_DENSITY_THRESHOLD = 0.1 # kilograms per cubic meter (kg/m^3)
    CONSUMPTION_RATE = 0.3  # kilograms per square meter per second (kg/(m^2·s)) - Surface area consumption rate

    # Boundary conditions
    BOUNDARY_FACTORS = {
        'west': 0.5, # Dimensionless (fraction)
        'east': 0.5, # Dimensionless (fraction)
        'south': 0.5, # Dimensionless (fraction)
        'north': 0.5, # Dimensionless (fraction)
        'top': 0.8 # Dimensionless (fraction)
    }
    WIND_UPDRAFT_FACTOR = 0.5 # Dimensionless (fraction)
    OXYGEN_FACTORS = {
        'surface': 1.0, # Dimensionless (relative fraction)
        'midstory': 0.8, # Dimensionless (relative fraction)
        'canopy': 0.6 # Dimensionless (relative fraction)
    }
    HEIGHT_THRESHOLDS = {
        'surface': 1.5, # meters (m)
        'midstory': 10.0 # meters (m)
    }
    PLUME_CONSTANT = 0.377 # Dimensionless (empirical constant)
    # PLUME_HEIGHT_FACTOR = 0.1 # meters per (kilowatt to the power of PLUME_POWER) (m/kW^PLUME_POWER)
    # PLUME_POWER = 0.6 # Dimensionless
    WIND_FIELD_SIZE_LIMIT = 100.0  # Megabytes (MB)

### ----- Procedural Environment Constants -----
    MAX_TERRAIN_ELEVATION = 50 # meters (m)
    BURN_BOX_PADDING = 10 # Pad burn box to allow agent access to surround fire
    IGNITION_INNER_BUFFER = 5
    NUM_WIND_UPDATES = 2
    NUM_IGNITION_LINES = 2

### ----- AI Constants -----

    # --- Hyperparameters ---
    DEFAULT_NUM_EPISODES = 1
    DEFAULT_AGENT_OBS_GRID_X = 20
    DEFAULT_AGENT_OBS_GRID_Y = 20
    DEFAULT_NUM_AGENTS = 4
    DEFAULT_LEARNING_RATE = 3e-4
    DEFAULT_GAMMA = 0.99
    DEFAULT_ENTROPY_BETA = 0.01
    DEFAULT_TERMINATION_BETA = 0.01
    DEFAULT_BUFFER_CAPACITY = 10000
    DEFAULT_BATCH_SIZE = 64
    CHECKPOINT_INTERVAL = 50

    # --- Action Capacities & Costs ---
    # Moisture and fire suppression
    MOISTURE_INCREASE_WATER = 0.2  # Dimensionless (fractional increase)
    MOISTURE_INCREASE_RETARDANT = 0.4 # Dimensionless (fractional increase)
    MOISTURE_INCREASE_AIR_SUPPORT = 0.6 # Dimensionless (fractional increase)
    
    # Costs are in units of resources (e.g., liters of water, kg of retardant). Each agent starts with a fixed amount of resources.
    WATER_COST = 1 # Dimensionless (resource unit)
    FIREBREAK_COST = 0 # Dimensionless (resource unit)
    RETARDANT_COST = 1 # Dimensionless (resource unit)
    AIR_SUPPORT_COST = 1 # Dimensionless (resource unit)

    # Resource capacities
    WATER_CAPACITY = 100
    RETARDANT_CAPACITY = 50
    AIR_SUPPORT_CAPACITY = 20

    # Action-specific radii/ranges
    WATER_ACTION_RANGE = 3      # Dimensionless (grid cells)
    FIREBREAK_ACTION_RANGE = 0  # Dimensionless (grid cells)
    RETARDANT_ACTION_RANGE = 3  # Dimensionless (grid cells)
    AIR_SUPPORT_ACTION_RANGE = 20 # Dimensionless (grid cells, effectively no range limit for target selection)

    WATER_ACTION_Z_RANGE = DEFAULT_NZ      # Dimensionless (grid cells)
    FIREBREAK_ACTION_Z_RANGE = DEFAULT_NZ  # Dimensionless (grid cells)
    RETARDANT_ACTION_Z_RANGE = DEFAULT_NZ  # Dimensionless (grid cells)
    AIR_SUPPORT_ACTION_Z_RANGE = DEFAULT_NZ # Dimensionless (grid cells, effectively no range limit for target selection)

    AIR_SUPPORT_GRID_RADIUS = 1 # Dimensionless (grid cells radius, e.g., 1 implies 3x3 area)
    NAVIGATION_SAFE_HEIGHT = 2 # Dimensionless (z-level index)

    IDLE_DELAY_STEPS = 3 # Number of steps action is delayed
    AIR_SUPPORT_DELAY_STEPS = 3 # Number of steps action is delayed
    
    # --- Rewards --- 
    PENALTY_ACTIVE_FIRE_OUTSIDE_BB = -100.0  # Large penalty for fire spread
    PENALTY_FUEL_CONSUMED_OUTSIDE_BB = -80.0  # Slightly less than fire penalty
    REWARD_FUEL_CONSUMED_INSIDE_BB = 80.0 # Encourage fire burn inside of burn box
    PENALTY_GLOBAL_WATER_USAGE = -5.0  # Moderate resource penalty
    PENALTY_GLOBAL_RETARDANT_USAGE = -7.0  # Higher for costlier resource
    PENALTY_GLOBAL_AIR_SUPPORT_USAGE = -10.0  # Highest for most expensive resource
    PENALTY_TIME_STEP = -0.5  # Small per-step penalty
    REWARD_MISSION_CONTAINED_SUCCESS = 100.0  # Large reward for success
    PENALTY_MISSION_FAILURE_TIMEOUT = -100.0  # Large penalty for timeout
    PENALTY_MISSION_FAILURE_SPREAD = -150.0  # Larger penalty for uncontrolled spread
    REWARD_AGENT_FIRE_REDUCTION = 5.0  # Reward for effective fire suppression
    PENALTY_RESOURCE_INVALID_ACTION = -2.0  # Penalty for invalid actions
    MAX_GLOBAL_REWARD_CLIP = 100.0  # Symmetric clipping
    MAX_AGENT_REWARD_CLIP = 10.0  # Smaller clipping for agents   

    # Movement cost
    MOVEMENT_PENALTY = -0.5 # Cost of moving one unit

    # Alpha for reward shaping
    DEFAULT_REWARD_ALPHA = 0.8 # Dimensionless (shaping factor between 0 and 1)
    
    # --- Win Condition Constants (Fuel-Based) ---
    WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD = 0.05 # Percent of total fuel allowed to burn and considered win

    # --- AI Pathfinding ---
    HEAT_FLUX_DANGER_THRESHOLD = 10 # kW acceptable to humans
    MOISTURE_PERCENTILE_THRESHOLD = 0.80 # Top Percentile of grid cells with moisture
    MAX_ESCAPE_SEARCH_COST = 20 # Tunable parameter governing how far out from agent's position to look for safe spot
    BURN_BOX_MOVEMENT_COST = 5 # Cost of moving across the burn box
    # --- AI Visualization ---
    VISUALIZATION_PLAYBACK_SPEED = 3 # FPS

    # --- Curriculum Manager ---
    ELO_INITIAL_AGENT = 1000
    ELO_K_FACTOR_AGENT = 32
    ELO_K_FACTOR_ENV = 16
    ELO_TARGET_DIFFERENCE = 50
    ELO_RANGE_SELECTION = 150

    BURN_IN_EPISODES = 500                # Total episodes for Phase 1
    BURN_IN_BATCH_SIZE = 50               # Episodes per Elo update during burn-in

    NUM_ENVIRONMENTS_PER_POOL = 100       # How many environments in each regenerated pool
    N_CALIBRATION_EPISODES = 10           # Episodes per environment during mini-burn-in (Phase 2.2)

    N_EPISODES_PER_ELO_UPDATE = 20        # Batch size for Elo updates in main curriculum (Phase 3.2, 3.4)
    N_REGENERATION_PERIOD_BATCHES = 50    # How many "batches" (of N_EPISODES_PER_ELO_UPDATE) before regenerating pool (Phase 3.5)

    RANDOM_REPLAY_PROB = 0.1              # Probability of triggering a random replay instead of curriculum selection (Phase 3.4)

    TOTAL_CURRICULUM_BATCHES = 500        # Total number of batches to simulate in Phase 3 after burn-in

    KEEP_TOP_N = 10                       # Number of "Parent" environments kept in the next population
    MUTATE_N = 10                         # Number of "Mutated" environments kept in the next population

    # --- Default Base Config ---
    BASE_CONFIG = {
        'nx': DEFAULT_NX,
        'ny': DEFAULT_NY,
        'nz': DEFAULT_NZ, 
        'dx': DEFAULT_DX, 
        'dy': DEFAULT_DY,
        'max_grid_height': DEFAULT_MAX_GRID_HEIGHT,
        'max_elevation': MAX_TERRAIN_ELEVATION,
        'random_seed': DEFAULT_RANDOM_SEED,
        'total_time': DEFUALT_SIMULATION_TIME, 
        'num_episodes': DEFAULT_NUM_EPISODES,
        'obs_grid_x': DEFAULT_AGENT_OBS_GRID_X, 
        'obs_grid_y': DEFAULT_AGENT_OBS_GRID_Y, 
        'num_agents': DEFAULT_NUM_AGENTS,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'gamma': DEFAULT_GAMMA,
        'entropy_beta': DEFAULT_ENTROPY_BETA,
        'termination_beta': DEFAULT_TERMINATION_BETA,
        'buffer_capacity': DEFAULT_BUFFER_CAPACITY,
        'batch_size': DEFAULT_BATCH_SIZE,
        'reward_alpha': DEFAULT_REWARD_ALPHA
    }