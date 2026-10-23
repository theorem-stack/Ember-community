import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Optional
from AI.FirefighterEnvironment import FirefighterEnvironment
from AI.OptionCriticNetwork import OptionCriticNetwork
from AI.OptionReplayBuffer import OptionReplayBuffer

class OptionCriticAgent:
    """
    Option-Critic agent for multi-agent firefighting environment.
    Learns a policy over options, and a termination condition.
    """
    def __init__(self, 
                 env: FirefighterEnvironment,
                 policy_net: Optional[OptionCriticNetwork] = None,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 entropy_beta: float = 0.01, # Coefficient for entropy regularization
                 termination_beta: float = 0.01, # Coefficient for termination regularization
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 reward_alpha: Optional[float] = None,
                 device: torch.device = None,
                 is_training: bool = True
        ):
        
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.termination_beta = termination_beta
        self.batch_size = batch_size
        if reward_alpha is None:
            from QuicFire.QuicFireConstants import QuicFireConstants
            reward_alpha = QuicFireConstants.DEFAULT_REWARD_ALPHA
        self.reward_alpha = reward_alpha # Store alpha
        self.verbose = True # Assuming you have a verbose flag
        self.device = device
        self.num_agents = env.num_agents

        # Initialize the Option-Critic Network and move it to the specified device
        obs_shape = env.observation_space.shape
        # Convert (H, W, C) to (C, H, W) for PyTorch Conv2d
        self.input_channels = obs_shape[2]
        grid_size_x = obs_shape[0]
        grid_size_y = obs_shape[1]
        num_action_types = env.num_action_types

        # Specify the policy network
        if policy_net:
            assert policy_net.input_channels == self.input_channels, "Input channels mismatch"
            assert policy_net.grid_size_x == grid_size_x, "Grid size x mismatch"
            assert policy_net.grid_size_y == grid_size_y, "Grid size y mismatch"
            assert policy_net.num_action_types == num_action_types, "Number of action types mismatch"
            self.policy_net = policy_net.to(self.device)
        else:
            self.policy_net = OptionCriticNetwork(self.input_channels, grid_size_x, grid_size_y, num_action_types).to(self.device)

        # Specify if training
        if is_training:
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            self.buffer = OptionReplayBuffer(buffer_capacity)
        else:
            self.policy_net.eval()  # Set to evaluation mode

        # Store agent-specific states for SMDP
        self.agent_initial_option_states: List[Optional[np.ndarray]] = [None] * self.num_agents
        self.agent_chosen_options: List[Optional[Dict]] = [None] * self.num_agents
        self.agent_accumulated_rewards: List[float] = [0.0] * self.num_agents
        self.agent_option_steps: List[int] = [0] * self.num_agents
        self.agent_initial_values: List[float] = [0.0] * self.num_agents
        self.agent_initial_option_type_log_probs: List[float] = [0.0] * self.num_agents
        self.agent_initial_target_xy_log_probs: List[float] = [0.0] * self.num_agents

    def select_option(self, state: np.ndarray, agent_idx: int) -> Dict:
        """
        Select an option (action_type, x, y) for an agent using Option-Critic policy.

        Args:
            state (np.ndarray): Current observation.
            agent_idx (int): Index of the agent.

        Returns:
            Dict: Selected option with action_type, x, y, and log probabilities.
        """
        # --- Change 1: Move input state to device before processing ---
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        masks = self.env.get_valid_actions_mask(agent_idx)
        # The masks are already created as Tensors with `to(self.device)`
        action_type_mask = torch.BoolTensor(masks['action_type_mask']).to(self.device)
        target_xy_mask_2d = torch.BoolTensor(masks['target_xy_mask_2d']).to(self.device)
        
        with torch.no_grad():
            option_type_logits, target_xy_logits, termination_logit, value = self.policy_net(state_tensor)

        # --- Change 2: Combine masks to prevent sampling an invalid action type ---
        # An action type is valid only if it is allowed by resource checks AND has at least one valid (x,y) target.
        has_valid_targets = target_xy_mask_2d.any(dim=(1, 2))
        
        # Create the final, combined mask for the option type
        final_option_type_mask = action_type_mask & has_valid_targets
        
        # --- Fallback logic for when there are NO valid actions at all ---
        chosen_option = {
            'agent_idx': agent_idx,
            'action_type': self.env.action_type_map['idle'],
            'x': self.env.agent_pos[agent_idx][0],
            'y': self.env.agent_pos[agent_idx][1],
            'option_type_log_prob': torch.tensor(0.0, device=self.device),
            'target_xy_log_prob': torch.tensor(0.0, device=self.device),
            'value': torch.tensor(0.0, device=self.device)
        }

        if not final_option_type_mask.any():
            if self.verbose:
                print(f"Agent {agent_idx}: No valid actions available, defaulting to idle.")
        else:
            # --- Apply the final, combined mask to option type logits ---
            option_type_logits = option_type_logits.masked_fill(~final_option_type_mask.unsqueeze(0), -float('inf'))
            option_type_probs = torch.softmax(option_type_logits, dim=1)
            option_type_dist = torch.distributions.Categorical(probs=option_type_probs)
            
            # Sample an action type (guaranteed to have valid targets)
            chosen_action_type = option_type_dist.sample()
            action_type_log_prob = option_type_dist.log_prob(chosen_action_type)

            # Select the 2D logits and mask for the chosen action type
            current_target_xy_logits = target_xy_logits[0, chosen_action_type] # Shape: (grid_size_x, grid_size_y)
            current_target_xy_mask_2d = target_xy_mask_2d[chosen_action_type] # Shape: (grid_size_x, grid_size_y)

            # Flatten logits and apply mask (guaranteed not to be all -inf)
            flattened_logits = current_target_xy_logits.view(-1)
            flattened_mask = current_target_xy_mask_2d.view(-1)
            flattened_logits_masked = flattened_logits.masked_fill(~flattened_mask, -float('inf'))

            # Compute probabilities and sample from the joint (x,y) distribution
            target_xy_probs = torch.softmax(flattened_logits_masked, dim=0)
            target_xy_dist = torch.distributions.Categorical(probs=target_xy_probs)
            
            chosen_xy_flat_idx = target_xy_dist.sample()
            target_xy_log_prob = target_xy_dist.log_prob(chosen_xy_flat_idx)

            # Convert flat index back to (x, y) coordinates
            chosen_x = chosen_xy_flat_idx // self.env.grid_size_y
            chosen_y = chosen_xy_flat_idx % self.env.grid_size_y
            
            chosen_option = {
                'agent_idx': agent_idx,
                'action_type': chosen_action_type.item(),
                'x': chosen_x.item(),
                'y': chosen_y.item(),
                'option_type_log_prob': action_type_log_prob,
                'target_xy_log_prob': target_xy_log_prob,
                'value': value.squeeze()
            }

        # Store chosen_option... (rest of the function is unchanged)
        self.agent_chosen_options[agent_idx] = {
            'action_type': chosen_option['action_type'],
            'x': chosen_option['x'],
            'y': chosen_option['y']
        }

        self.agent_initial_option_states[agent_idx] = state
        self.agent_initial_values[agent_idx] = chosen_option['value'].item()
        self.agent_initial_option_type_log_probs[agent_idx] = chosen_option['option_type_log_prob'].item()
        self.agent_initial_target_xy_log_probs[agent_idx] = chosen_option['target_xy_log_prob'].item()

        return chosen_option

    def update(self):
        """
        Performs a training update using a batch of option trajectories from the buffer.
        Calculates Option-Critic specific losses (policy, termination, value).
        """
        if len(self.buffer) < self.batch_size:
            # print(f"Buffer size {len(self.buffer)} is less than batch size {self.batch_size}. Skipping update.")
            return

        batch = self.buffer.sample(self.batch_size)

        # --- Change 2: Move all tensors from the buffer to the device ---
        initial_states = torch.FloatTensor(np.array([t['initial_state'] for t in batch])).permute(0, 3, 1, 2).to(self.device)
        final_states = torch.FloatTensor(np.array([t['final_state'] for t in batch])).permute(0, 3, 1, 2).to(self.device)
        rewards = torch.FloatTensor([t['accumulated_reward'] for t in batch]).to(self.device)
        steps_taken = torch.FloatTensor([t['steps_taken'] for t in batch]).to(self.device)
        initial_values = torch.FloatTensor([t['initial_value'] for t in batch]).to(self.device)
        initial_option_type_log_probs = torch.FloatTensor([t['initial_option_type_log_prob'] for t in batch]).to(self.device)
        initial_target_xy_log_probs = torch.FloatTensor([t['initial_target_xy_log_prob'] for t in batch]).to(self.device)
        final_termination_probs = torch.FloatTensor([t['final_termination_prob'] for t in batch]).to(self.device)

        # Get network outputs for initial and final states
        initial_option_type_logits, initial_target_xy_logits, initial_termination_logit, initial_V = self.policy_net(initial_states)
        final_option_type_logits, final_target_xy_logits, final_termination_logit, final_V = self.policy_net(final_states)

        # Calculate Q-values for the chosen options (Approximation as before)
        target_V = rewards + (self.gamma ** steps_taken) * (
            (1 - final_termination_probs) * final_V.squeeze() + final_termination_probs * final_V.squeeze()
        )
        
        # Value Loss (MSE between predicted V and target V)
        value_loss = F.mse_loss(initial_V.squeeze(), target_V.detach())

        # Option Policy Loss
        # We need to get the log_prob of the *chosen* option from the initial state's logits.
        # --- Change 3: Move chosen option indices to device ---
        chosen_option_types = torch.LongTensor([t['chosen_option']['action_type'] for t in batch]).to(self.device)
        chosen_target_x = torch.LongTensor([t['chosen_option']['x'] for t in batch]).to(self.device)
        chosen_target_y = torch.LongTensor([t['chosen_option']['y'] for t in batch]).to(self.device)

        # Get log_probs for the chosen action type
        current_option_type_log_probs = F.log_softmax(initial_option_type_logits, dim=-1).gather(1, chosen_option_types.unsqueeze(-1)).squeeze(-1)
        
        # Flatten the (x,y) coordinates to a single index
        grid_size_y = self.env.grid_size_y
        chosen_xy_flat_indices = chosen_target_x * grid_size_y + chosen_target_y

        # Reshape initial_target_xy_logits for gather operation (batch_size, num_action_types, grid_size_x * grid_size_y)
        initial_target_xy_logits_flat = initial_target_xy_logits.view(self.batch_size, self.env.num_action_types, -1)
        
        # Apply log_softmax for each action type over the flattened (x,y) dimensions
        log_probs_xy_for_each_action_type = F.log_softmax(initial_target_xy_logits_flat, dim=-1)

        # Now, gather the log_prob for the *specifically chosen* (action_type, x, y) combination
        current_target_xy_log_probs = log_probs_xy_for_each_action_type.gather(
            1, chosen_option_types.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.env.grid_size_x * self.env.grid_size_y)
        ).squeeze(1).gather(1, chosen_xy_flat_indices.unsqueeze(1)).squeeze(1)

        # Total log probability of the chosen option
        total_option_log_prob = current_option_type_log_probs + current_target_xy_log_probs

        # Advantage for policy update
        advantage = (target_V - initial_V.squeeze()).detach()
        policy_loss = -(total_option_log_prob * advantage).mean()

        # Entropy Loss for exploration
        option_type_entropy = Categorical(logits=initial_option_type_logits).entropy().mean()
        
        # Need to get the entropy for each action type's (x,y) distribution
        initial_target_xy_probs_flat = torch.softmax(initial_target_xy_logits.view(self.batch_size, self.env.num_action_types, -1), dim=-1)
        
        # Calculate entropy for each (action_type, batch_item)
        target_xy_entropy = 0
        for i in range(self.batch_size):
            for j in range(self.env.num_action_types):
                target_xy_entropy += Categorical(probs=initial_target_xy_probs_flat[i, j]).entropy()
        target_xy_entropy = target_xy_entropy / (self.batch_size * self.env.num_action_types)

        entropy_loss = -(self.entropy_beta * (option_type_entropy + target_xy_entropy))

        # Termination Loss
        initial_termination_prob = torch.sigmoid(initial_termination_logit)
        termination_loss = -((1 - initial_termination_prob.squeeze()) * advantage).mean()
        termination_loss -= self.termination_beta * (initial_termination_prob * (1 - initial_termination_prob)).mean()

        # Total Loss
        total_loss = policy_loss + value_loss + entropy_loss + termination_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

    def select_and_execute_options(self, env: FirefighterEnvironment, state: np.ndarray, active_flags: List[bool]) -> List[List[int]]:
        """Selects or continues options for all agents."""
        actions = []
        for agent_idx in range(env.num_agents):
            if not active_flags[agent_idx] or env.option_terminated[agent_idx]:
                # The state passed to select_option is a numpy array. The function will handle moving it to the device.
                option = self.select_option(state, agent_idx) 
                actions.append([option['agent_idx'], option['action_type'], option['x'], option['y']])
                active_flags[agent_idx] = True
            else:
                intent = env.agent_target_intent[agent_idx]
                if intent:
                    actions.append([agent_idx, intent['type'], intent['x'], intent['y']])
                else:
                    active_flags[agent_idx] = False
                    pos = env.agent_pos[agent_idx]
                    actions.append([agent_idx, env.action_type_map['firebreak'], pos[0], pos[1]])
        return actions
    
    def update_agent_tracking(self, env: FirefighterEnvironment, next_state: np.ndarray,
                          rewards: List[float], global_reward: float, active_flags: List[bool]) -> None:
        """Updates agent tracking and stores completed trajectories."""
        for agent_idx in range(env.num_agents):
            # Only process agents that are currently active
            if active_flags[agent_idx]:
                blended_reward = (self.reward_alpha * rewards[agent_idx]) + ((1 - self.reward_alpha) * global_reward)
                self.agent_accumulated_rewards[agent_idx] += blended_reward
                self.agent_option_steps[agent_idx] += 1

                if env.option_terminated[agent_idx]:
                    # --- CORRECTED: Add an explicit check for initial_state validity ---
                    if self.agent_initial_option_states[agent_idx] is not None:
                        final_state_tensor = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        _, _, final_termination_logit, _ = self.policy_net(final_state_tensor)
                        final_termination_prob = torch.sigmoid(final_termination_logit).item()

                        self.buffer.store_option_trajectory(
                            initial_state=self.agent_initial_option_states[agent_idx],
                            chosen_option=self.agent_chosen_options[agent_idx],
                            accumulated_reward=self.agent_accumulated_rewards[agent_idx],
                            steps_taken=self.agent_option_steps[agent_idx],
                            final_state=next_state,
                            initial_value=self.agent_initial_values[agent_idx],
                            initial_option_type_log_prob=self.agent_initial_option_type_log_probs[agent_idx],
                            initial_target_xy_log_prob=self.agent_initial_target_xy_log_probs[agent_idx],
                            final_termination_prob=final_termination_prob
                        )
                    else:
                        # Log a warning to help with future debugging
                        print(f"Warning: Agent {agent_idx} terminated but initial state was None. Trajectory not stored.")

                    # Reset tracking for this specific agent
                    active_flags[agent_idx] = False
                    self.agent_initial_option_states[agent_idx] = None
                    self.agent_chosen_options[agent_idx] = None
                    self.agent_accumulated_rewards[agent_idx] = 0.0
                    self.agent_option_steps[agent_idx] = 0
                    self.agent_initial_values[agent_idx] = 0.0
                    self.agent_initial_option_type_log_probs[agent_idx] = 0.0
                    self.agent_initial_target_xy_log_probs[agent_idx] = 0.0