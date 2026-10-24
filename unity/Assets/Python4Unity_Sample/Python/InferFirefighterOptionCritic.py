import argparse
import os
import sys
import json
import numpy as np
import torch
import onnxruntime as ort
from AI.FirefighterEnvironment import FirefighterEnvironment
from AI.OptionCriticAgent import OptionCriticAgent
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireTuning import apply_document
from QuicFire.RNGCollection import RNGCollection
from QuicFire.QuicFireEnvGenerator import ClimateConditions
from typing import Dict, Tuple

# Global variables
_env = None
_agent = None
_device = None
_outputFilePath = None
_episode_reward = None
_done = None
_agent_option_active = None
_next_state = None

class ONNXPolicyWrapper(torch.nn.Module):
    """
    A PyTorch module that wraps an ONNX session to mimic the agent's policy network.
    It stores the key network attributes for compatibility.
    """
    def __init__(self, ort_session: ort.InferenceSession, 
                    input_channels: int, 
                    grid_size_x: int, 
                    grid_size_y: int, 
                    num_action_types: int):
        super().__init__()
        self.ort_session = ort_session
        
        # --- Add the metadata attributes for compatibility ---
        self.input_channels = input_channels
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_action_types = num_action_types

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ONNX inference and return outputs as PyTorch tensors."""
        # Convert PyTorch tensor to NumPy for ONNX
        ort_inputs = {self.ort_session.get_inputs()[0].name: x.cpu().numpy()}
        
        # Run inference
        ort_outputs = self.ort_session.run(None, ort_inputs)
        
        # Convert outputs to PyTorch tensors
        option_logits = torch.from_numpy(ort_outputs[0]).to(x.device)
        target_xy_logits = torch.from_numpy(ort_outputs[1]).to(x.device)
        
        # --- CORRECTED: The ONNX model should output all 4 logits from the NN ---
        termination_logit = torch.from_numpy(ort_outputs[2]).to(x.device)
        value = torch.from_numpy(ort_outputs[3]).to(x.device)
        
        return option_logits, target_xy_logits, termination_logit, value
    
def load_onnx_model(onnx_model_path: str, config_data: Dict) -> ONNXPolicyWrapper:
    """
    Loads an ONNX model and its configuration into a wrapped policy network.
    """
    
    input_channels = config_data['num_input_channels']
    grid_size_x = config_data['obs_grid_x']
    grid_size_y = config_data['obs_grid_y']
    num_action_types = config_data['num_action_types']
    
    # 2. Create the ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # 3. Instantiate the wrapper with the loaded metadata
    onnx_policy_net = ONNXPolicyWrapper(
        ort_session=ort_session,
        input_channels=input_channels,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        num_action_types=num_action_types
    )
    
    return onnx_policy_net

def initialize_environment(config: Dict, climate: ClimateConditions, module_seeds: RNGCollection) -> FirefighterEnvironment:
    """Initialize the FirefighterEnvironment with given configuration."""
    env_max_steps = max(int(config['total_time'] / QuicFireConstants.DT), 1)
    return FirefighterEnvironment(
        grid_size=(config['obs_grid_x'], config['obs_grid_y']),
        max_steps=env_max_steps,
        num_agents=config['num_agents'],
        sim_base_config=config,
        module_seeds=module_seeds,
        climate=climate,
        verbose=False,
        enable_viz_recorder=False,
        enable_reward_logging=False,
        is_training=False
    )

def setup_environment(config_filename: str ="firefighter_config.json", outputFilePath: str ="./OutputData/viz_data.json") -> Dict:
    """Run inference using an ONNX model with the OptionCriticAgent."""
    print(f"Using onnxruntime version: {ort.__version__}", file=sys.stderr)

    global _env

    # Get Full Paths Relative to Python Root
    current_file_path = os.path.dirname(__file__)
    config_path = os.path.join(current_file_path, config_filename)

    # 1. Load the configuration file to get model parameters.
    # Apply allowlisted constants before total_time / DT.
    with open(config_path, 'r') as f:
        document = json.load(f)
    apply_document(document)
    config_data = document['config']
    
    # Initialize environment
    climate = ClimateConditions(temperature=260, humidity=0.6, precipitation=600.0)
    module_seeds = RNGCollection(config_data['random_seed'])
    
    _env = initialize_environment(config_data, climate, module_seeds)

    # Create Serializable Object
    environment_data = {
        'height_map': _env.height_map.tolist(),
        'veg_map': _env.veg_map.tolist(),
        'fuel_map': _env.fuel_map.tolist(),
        'fuel_moisture_map': _env.fuel_moisture_map.tolist(),
        'burn_box_coords': _env.sim.burn_box_coords,
        'burn_box_mask': _env.burn_box_obs_mask.tolist(),
        'grid_metadata': {
            'nx': _env.sim_base_config['nx'],
            'ny': _env.sim_base_config['ny'],
            'dx': _env.sim_base_config['dx'],
            'dy': _env.sim_base_config['dy'],
            'z_levels': _env.sim.z_levels.tolist(),
            'dz': _env.sim.dz.tolist(),
            'agent_obs_x': _env.grid_size_x, 
            'agent_obs_y': _env.grid_size_y
        }
    }

    # Write the JSON data to a file
    with open(outputFilePath, 'w') as f:
        json.dump(environment_data, f)

    return "setup_environment"

def setup_agent(model_filename: str = "firefighter_policy.onnx", config_filename: str ="firefighter_config.json"):

    global _env, _agent, _device

    # Get Full Paths Relative to Python Root
    current_file_path = os.path.dirname(__file__)
    model_path = os.path.join(current_file_path, model_filename)
    config_path = os.path.join(current_file_path, config_filename)

    # 1. Load the configuration file to get model parameters.
    # apply_document is idempotent if setup_environment already ran.
    with open(config_path, 'r') as f:
        document = json.load(f)
    apply_document(document)
    config_data = document['config']

    # Replace agent's policy_net with ONNX wrapper
    policy_wrapper = load_onnx_model(model_path, config_data)
    
    # Set device to CPU for compatibility with Unity/Nuitka
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _device = torch.device("cpu")

    # Initialize agent for inference (no learning)
    _agent = OptionCriticAgent(
        _env,
        policy_net=policy_wrapper,
        learning_rate=0.0,  # Disable learning
        gamma=config_data.get('gamma', 0.99),
        entropy_beta=0.0,
        termination_beta=0.0,
        buffer_capacity=1,  # Minimal buffer
        batch_size=1,
        reward_alpha=config_data.get('reward_alpha', QuicFireConstants.DEFAULT_REWARD_ALPHA),
        device=_device,
        is_training=False
    )

    return "setup_agent"

def setup_episode(outputFilePath: str):

    global _env, _agent, _device, _outputFilePath, _episode_reward, _done, _agent_option_active, _next_state

    _outputFilePath = outputFilePath

    _episode_reward = 0.0
    _done = False
    _agent_option_active = [False] * _env.num_agents
    _next_state = _env._get_observation()
    
    # Store agent-specific states for SMDP
    _agent.agent_initial_option_states = [None] * _env.num_agents
    _agent.agent_chosen_options = [None] * _env.num_agents
    _agent.agent_accumulated_rewards = [0.0] * _env.num_agents
    _agent.agent_option_steps = [0] * _env.num_agents
    _agent.agent_initial_values = [0.0] * _env.num_agents
    _agent.agent_initial_option_type_log_probs = [0.0] * _env.num_agents
    _agent.agent_initial_target_xy_log_probs = [0.0] * _env.num_agents

    return "epsisode_setup_successful"

def step_episode():

    global _env, _agent, _device, _outputFilePath, _episode_reward, _done, _agent_option_active, _next_state

    with torch.no_grad():
        # Use the agent's existing method with ONNX-backed policy_net
        actions = _agent.select_and_execute_options(_env, _next_state, _agent_option_active)
    
    # Step the environment
    _next_state, rewards, _done, info = _env.step(np.array(actions, dtype=int))
    
    # Stream visualization data for Unity
    viz_data = info.get('viz_data', {})

    # Write the JSON data to a file
    with open(_outputFilePath, 'w') as f:
        json.dump(viz_data, f)
    
    # Update state and reward
    global_reward = info.get('global_reward', 0.0)
    _episode_reward += global_reward
    
    if _done and info.get('status') == 'failed':
        return "episode_step_failed"
    
    return "episode_step_successful"

def get_status():
    return _done

DEFAULT_MODEL_PATH = "TrainedModels/firefighter_policy_batch_0_20250804_143554.onnx"
DEFAULT_CONFIG_PATH = "TrainedModels/firefighter_policy_config_20250804_143517.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ONNX firefighter inference against a trained checkpoint."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="ONNX policy. Relative paths are resolved from this script (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Trainer config JSON. Relative paths are resolved from this script (default: %(default)s).",
    )
    return parser


def checkpoint_path(filename: str) -> str:
    if os.path.isabs(filename):
        return filename
    return os.path.join(os.path.dirname(__file__), filename)


def require_checkpoint(path: str, label: str) -> None:
    """Exit with a clear message when a model or config file is missing."""
    if os.path.isfile(path):
        return
    print(
        f"Cannot start inference: {label} not found at {path}. "
        "Train a checkpoint with TrainFirefighterOptionCritic.py, or pass --model and --config.",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    model_path = checkpoint_path(args.model)
    config_path = checkpoint_path(args.config)
    require_checkpoint(model_path, "model")
    require_checkpoint(config_path, "config")

    output_filepath = "UnityVisuals/"
    current_file_path = os.path.dirname(__file__)
    folder_path = os.path.join(current_file_path, output_filepath)
    os.makedirs(folder_path, exist_ok=True)

    environment_output = os.path.join(folder_path, "environment_data.json")
    episode_output = os.path.join(folder_path, "episode_data.json")

    setup_environment(config_path, environment_output)
    setup_agent(model_path, config_path)
    setup_episode(episode_output)

    # Walk the episode. The repository-root copy takes a single step.
    while not _done:
        step_episode()
        print("step")
