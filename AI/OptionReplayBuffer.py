import numpy as np
from typing import List, Dict
import random
from collections import deque

class OptionReplayBuffer:
    """
    Replay buffer to store full option trajectories for Option-Critic training.
    Each entry stores data for a completed option.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store_option_trajectory(self,
                                initial_state: np.ndarray,
                                chosen_option: Dict, # {'agent_idx', 'action_type', 'x', 'y'}
                                accumulated_reward: float,
                                steps_taken: int,
                                final_state: np.ndarray,
                                initial_value: float, # V(s_initial)
                                initial_option_type_log_prob: float,
                                initial_target_xy_log_prob: float,
                                final_termination_prob: float # beta(s_final, o)
                                ):
        """
        Stores a completed option's trajectory in the buffer.
        """
        self.buffer.append({
            'initial_state': initial_state,
            'chosen_option': chosen_option,
            'accumulated_reward': accumulated_reward,
            'steps_taken': steps_taken,
            'final_state': final_state,
            'initial_value': initial_value,
            'initial_option_type_log_prob': initial_option_type_log_prob,
            'initial_target_xy_log_prob': initial_target_xy_log_prob,
            'final_termination_prob': final_termination_prob
        })

    def sample(self, batch_size: int) -> List[Dict]:
        """Samples a batch of option trajectories from the buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer) # Return all if not enough
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        """Clears all trajectories from the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)