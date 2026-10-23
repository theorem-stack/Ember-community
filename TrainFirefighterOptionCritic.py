import os
import numpy as np
import torch
from typing import Dict, Optional
from datetime import datetime
import json
from AI.FirefighterEnvironment import FirefighterEnvironment
from AI.OptionCriticAgent import OptionCriticAgent
from AI.PolicyExporter import PolicyExporter
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireTuning import apply_tuning
from QuicFire.RNGCollection import RNGCollection
from QuicFire.QuicFireEnvGenerator import ClimateConditions

def main_training(save_path: str = "./Firefighter_Models", tuning: Optional[Dict] = None) -> Dict:
    """Main entry point for training the firefighter RL agent using Option-Critic."""
    # Apply before total_time / DT inside initialize_environment.
    if tuning:
        apply_tuning(tuning)
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base configuration
    base_config = QuicFireConstants.BASE_CONFIG
    
    # Initialize climate and seeds
    climate = ClimateConditions(temperature=260, humidity=0.6, precipitation=600.0)
    module_seeds = RNGCollection(base_config['random_seed'])
    print(f"Module seeds: {module_seeds.module_seeds}")
    
    # Initialize environment and agent
    env = initialize_environment(base_config, climate, module_seeds)
    agent = initialize_agent(env, base_config, device)
    
    # Run training
    result = run_training_loop(env, agent, base_config, save_path)
    print(f"Training completed with status: {result['status']}, Final reward: {result['reward']:.4f}")
    
    return result

def initialize_environment(config: Dict, climate: ClimateConditions, module_seeds: RNGCollection) -> FirefighterEnvironment:
    """Initializes the FirefighterEnvironment with the provided configuration."""
    env_max_steps = max(int(config['total_time'] / QuicFireConstants.DT), 1)
    return FirefighterEnvironment(
        grid_size=(config['obs_grid_x'], config['obs_grid_y']),
        max_steps=env_max_steps,
        num_agents=config['num_agents'],
        sim_base_config=config,
        module_seeds=module_seeds,
        climate=climate,
        verbose=False,
        enable_viz_recorder=True,
        enable_reward_logging=True,
        is_training=True
    )

def initialize_agent(env: FirefighterEnvironment, hyperparameters: Dict, device: torch.device) -> OptionCriticAgent:
    """Initializes the OptionCriticAgent with the environment and hyperparameters."""
    return OptionCriticAgent(
        env,
        policy_net=None,
        learning_rate=hyperparameters.get('learning_rate', 3e-4),
        gamma=hyperparameters.get('gamma', 0.99),
        entropy_beta=hyperparameters.get('entropy_beta', 0.01),
        termination_beta=hyperparameters.get('termination_beta', 0.01),
        buffer_capacity=hyperparameters.get('buffer_capacity', 10000),
        batch_size=hyperparameters.get('batch_size', 64),
        reward_alpha=hyperparameters.get('reward_alpha', QuicFireConstants.DEFAULT_REWARD_ALPHA),
        device=device,
        is_training=True
    )

def run_training_loop(env: FirefighterEnvironment, agent: OptionCriticAgent, config: Dict, save_path: str) -> Dict:
    """Executes the training loop for the specified number of episodes."""
    num_episodes = config['num_episodes']
    best_reward = float('-inf')

    # Update and save config for training run
    config['num_action_types'] = env.num_action_types
    config['num_input_channels'] = len(env.observation_channels)
    save_config_to_file(config, save_path, "firefighter_policy_config")
    
    for episode in range(num_episodes):
        state = env.reset() if episode > 0 else env._get_observation()
        episode_result = run_episode(env, agent, state)
        
        # Logging and visualization
        if env.enable_viz_recorder:
            env.viz_recorder.animate_replay(output_filepath=None, fps=QuicFireConstants.VISUALIZATION_PLAYBACK_SPEED)
            env.viz_recorder.plot_action_locations('firebreak')
        
        if env.enable_reward_logging:
            # Conditional import
            from AI.Debugging.RewardVisualizer import RewardVisualizer

            visualizer = RewardVisualizer(episode_reward_log=env.episode_reward_log, num_agents=env.num_agents)
            visualizer.plot_all(save_dir=os.path.join(save_path, "reward_plots"))
        
        # Update agent
        agent.update()
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_result['reward']:.4f}, Steps: {episode_result['steps']}")
        save_episode_reward(save_path, "firefighter_log.txt", episode, episode_result['reward'])
        
        # Save checkpoint periodically or on best reward
        if episode_result['reward'] > best_reward or episode % 10 == 0 or episode == num_episodes - 1:
            if episode_result['reward'] > best_reward:
                best_reward = episode_result['reward']
                print(f"New best reward: {best_reward:.4f}")

            save_checkpoint(config, agent, save_path, "firefighter_policy", episode)
        
        if episode_result['status'] == 'failed':
            print(f"Training failed: {episode_result['reason']}")
            return episode_result
    
    return {'episode_idx': num_episodes - 1, 'status': 'completed', 'reward': episode_result['reward'], 'best_reward': best_reward}

def run_episode(env: FirefighterEnvironment, agent: OptionCriticAgent, initial_state: np.ndarray) -> Dict:
    """Runs a single training episode."""
    episode_reward = 0.0
    done = False
    agent_option_active = [False] * env.num_agents
    
    current_state = initial_state

    # Reset agent tracking
    agent.agent_initial_option_states = [None] * env.num_agents
    agent.agent_chosen_options = [None] * env.num_agents
    agent.agent_accumulated_rewards = [0.0] * env.num_agents
    agent.agent_option_steps = [0] * env.num_agents
    agent.agent_initial_values = [0.0] * env.num_agents
    agent.agent_initial_option_type_log_probs = [0.0] * env.num_agents
    agent.agent_initial_target_xy_log_probs = [0.0] * env.num_agents
    
    while not done:
        actions = agent.select_and_execute_options(env, current_state, agent_option_active)
        next_state, rewards, done, info = env.step(np.array(actions, dtype=int))
        
        if done and info.get('status') == 'failed':
            return {
                'status': 'failed',
                'reason': info.get('reason', 'no_active_fires'),
                'reward': 0.0,
                'steps': env.current_step
            }
        
        global_reward = info.get('global_reward', 0.0)
        episode_reward += global_reward
        agent.update_agent_tracking(env, next_state, rewards, global_reward, agent_option_active)
        current_state = next_state
    
    return {'status': 'completed', 'reward': episode_reward, 'steps': env.current_step}

def save_config_to_file(config: Dict, filepath: str, filename: str) -> None:
    """Saves training configuration to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not filename.endswith('.json'):
        filename = f"{filename}_{timestamp}.json"
    
    full_path = os.path.join(filepath, filename)
    
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    
    quic_fire_constants_dict = {
        key: _make_serializable(value) for key, value in vars(QuicFireConstants).items()
        if not key.startswith('__')
    }
    
    training_config = {
        'config': {k: _make_serializable(v) for k, v in config.items()},
        'quicfire_constants': quic_fire_constants_dict,
        'timestamp': timestamp
    }
    
    os.makedirs(filepath, exist_ok=True)
    try:
        with open(full_path, 'w') as f:
            json.dump(training_config, f, indent=4)
        print(f"Config saved: {full_path}")
    except IOError as e:
        print(f"Failed to save config to {full_path}: {e}")
        raise

def save_checkpoint(config: Dict, agent: OptionCriticAgent, folder_name: str, filename: str, batch_idx: int) -> None:
    """Saves the agent's policy network as an ONNX file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not filename.endswith('.onnx'):
        filename = f"{filename}_batch_{batch_idx}_{timestamp}.onnx"
    
    full_path = os.path.join(folder_name, filename)
    
    required_keys = ['obs_grid_x', 'obs_grid_y']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in config")
    
    input_dummy_shape = (agent.input_channels, config['obs_grid_x'], config['obs_grid_y'])
    
    os.makedirs(folder_name, exist_ok=True)
    try:
        PolicyExporter.export_to_onnx(agent.policy_net, input_dummy_shape, full_path)
        print(f"Checkpoint saved: {full_path}")
    except Exception as e:
        print(f"Failed to save checkpoint to {full_path}: {e}")
        raise

def save_episode_reward(save_path: str, file_name: str, episode: int, reward: float, append: bool = True) -> None:
    """Saves episode reward to a text file."""
    if not isinstance(save_path, str) or not isinstance(file_name, str):
        raise TypeError("save_path and file_name must be strings")
    if not isinstance(episode, int):
        raise TypeError("episode must be an integer")
    if not isinstance(reward, (int, float)):
        raise TypeError("reward must be a number")
    
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    line = f"{episode},{reward}\n"
    header = "episode,reward\n"
    
    file_exists = os.path.exists(file_path)
    mode = 'a' if append and file_exists else 'w'
    
    try:
        with open(file_path, mode) as f:
            if mode == 'w' or (append and not file_exists):
                f.write(header)
            f.write(line)
    except OSError as e:
        raise OSError(f"Error writing to file {file_path}: {e}")

if __name__ == "__main__":

    folder_name = "Training/"

    # Create full path
    current_file_path = os.path.dirname(__file__)
    folder_path = os.path.join(current_file_path, folder_name)

    main_training(save_path=folder_path)