import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os

class RewardVisualizer:
    """A class to visualize rewards from a FirefighterEnvironment episode."""
    
    def __init__(self, episode_reward_log: List[Dict[str, Any]], num_agents: int):
        """
        Initialize the visualizer with episode reward data.
        
        Args:
            episode_reward_log: List of dictionaries containing reward data per step.
            num_agents: Number of agents in the environment.
        """
        self.episode_reward_log = episode_reward_log
        self.num_agents = num_agents
        self.steps = [entry['step'] for entry in episode_reward_log]
        
        if not episode_reward_log:
            raise ValueError("episode_reward_log is empty. No data to visualize.")
        
    def plot_global_reward(self, figsize: tuple = (10, 6), save_path: str = None):
        """
        Plot a line chart of the global reward over episode steps.
        
        Args:
            figsize: Tuple of (width, height) for the figure size.
            save_path: Optional file path to save the plot.
        """
        global_rewards = [entry['global_reward'] for entry in self.episode_reward_log]
        
        plt.figure(figsize=figsize)
        plt.plot(self.steps, global_rewards, label='Global Reward', color='blue', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Global Reward')
        plt.title('Global Reward Over Episode')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_agent_total_rewards(self, figsize: tuple = (10, 6), save_path: str = None):
        """
        Plot a line chart of each agent's total reward over episode steps.
        
        Args:
            figsize: Tuple of (width, height) for the figure size.
            save_path: Optional file path to save the plot.
        """
        agent_rewards = [
            [entry['agent_rewards'][i]['total_reward'] for entry in self.episode_reward_log]
            for i in range(self.num_agents)
        ]
        
        plt.figure(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        for i in range(self.num_agents):
            plt.plot(self.steps, agent_rewards[i], label=f'Agent {i+1}', color=colors[i], linewidth=2)
        
        plt.xlabel('Step')
        plt.ylabel('Total Agent Reward')
        plt.title('Total Reward per Agent Over Episode')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_agent_reward_components(self, figsize: tuple = (10, 6), save_path: str = None):
        """
        Plot a stacked bar chart of total reward per agent, with positive rewards above
        and negative penalties below the x-axis.
        
        Args:
            figsize: Tuple of (width, height) for the figure size.
            save_path: Optional file path to save the plot.
        """
        # Reward and penalty components
        reward_components = ['fire_reduction']
        penalty_components = ['movement_cost', 'invalid_action_penalty']
        all_components = reward_components + penalty_components
        component_labels = ['Fire Reduction', 'Movement Cost', 'Invalid Action Penalty']
        colors = ['#1f77b4', '#ff7f0e', '#d62728']
        
        # Calculate total components for each agent over the episode
        total_components = {
            comp: [sum(entry['agent_rewards'][agent_idx].get(comp, 0.0)
                       for entry in self.episode_reward_log)
                   for agent_idx in range(self.num_agents)]
            for comp in all_components
        }
        
        # Log component values for debugging
        for comp in all_components:
            print(f"{comp}: {total_components[comp]}")
        
        # Calculate total reward per agent for reference
        total_rewards = [
            sum(entry['agent_rewards'][agent_idx]['total_reward']
                for entry in self.episode_reward_log)
            for agent_idx in range(self.num_agents)
        ]
        
        # Agent indices for x-axis
        agent_indices = np.arange(self.num_agents)
        
        # Plot stacked bar chart
        plt.figure(figsize=figsize)
        
        # Plot positive components (stacked above y=0)
        bottoms_positive = np.zeros(self.num_agents)
        for comp_idx, comp in enumerate(reward_components):
            values = np.array(total_components[comp])
            plt.bar(agent_indices, values, bottom=bottoms_positive,
                    label=component_labels[comp_idx], color=colors[comp_idx], alpha=0.8)
            bottoms_positive += np.where(values > 0, values, 0)
        
        # Plot negative components (stacked below y=0)
        bottoms_negative = np.zeros(self.num_agents)
        for comp_idx, comp in enumerate(penalty_components, start=len(reward_components)):
            values = np.array(total_components[comp])
            plt.bar(agent_indices, values, bottom=bottoms_negative,
                    label=component_labels[comp_idx], color=colors[comp_idx], alpha=0.8)
            bottoms_negative += np.where(values < 0, values, 0)
        
        # Plot total reward as a marker
        plt.scatter(agent_indices, total_rewards, color='black', marker='o', s=100,
                    label='Total Reward', zorder=5)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Agent')
        plt.ylabel('Total Reward / Penalty')
        plt.title('Total Reward per Agent by Component')
        plt.xticks(agent_indices, [f'Agent {i+1}' for i in range(self.num_agents)])
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_all(self, save_dir: str = None):
        """
        Generate all plots.
        
        Args:
            save_dir: Optional directory to save the plots.
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.plot_global_reward(save_path=os.path.join(save_dir, 'global_reward.png') if save_dir else None)
        self.plot_agent_total_rewards(save_path=os.path.join(save_dir, 'agent_total_rewards.png') if save_dir else None)
        self.plot_agent_reward_components(save_path=os.path.join(save_dir, 'agent_reward_components.png') if save_dir else None)

if __name__ == "__main__":
    # Example usage with dummy data (realistic positive/negative values)
    dummy_log = [
        {
            'step': i,
            'global_reward': 10 - i * 0.5,
            'global_reward_components': {
                'total_reward': 10 - i * 0.5,
                'active_fire_outside_bb_penalty': -i * 0.1,
                'fuel_consumed_outside_bb_penalty': -i * 0.05,
                'water_usage_penalty': -0.2,
                'retardant_usage_penalty': -0.1,
                'air_support_usage_penalty': -0.05,
                'time_penalty': -0.1,
                'mission_success_reward': 0.0 if i < 9 else 5.0,
                'mission_failure_spread_penalty': 0.0,
                'mission_failure_timeout_penalty': 0.0,
            },
            'agent_rewards': [
                {
                    'total_reward': 3 - i * 0.2,  # Net reward after components
                    'fire_reduction': 2.0 if i % 2 == 0 else 0.0,  # Positive, intermittent
                    'movement_cost': -0.5,  # Negative, every step
                    'invalid_action_penalty': -0.1 if i % 4 == 0 else 0.0,  # Negative, occasional
                },
                {
                    'total_reward': 2.5 - i * 0.15,
                    'fire_reduction': 1.0 if i % 2 == 1 else 0.0,
                    'movement_cost': -0.4,
                    'invalid_action_penalty': -0.05 if i % 5 == 0 else 0.0,
                }
            ]
        } for i in range(10)
    ]
    
    visualizer = RewardVisualizer(episode_reward_log=dummy_log, num_agents=2)
    visualizer.plot_all(save_dir="reward_plots")