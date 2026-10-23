import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from AI.FirefighterEnvironment import FirefighterEnvironment
from AI.OptionCriticAgent import OptionCriticAgent, save_config_to_file, save_checkpoint
from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireTuning import apply_tuning
from QuicFire.RNGCollection import RNGCollection
from QuicFire.QuicFireEnvGenerator import ClimateConditions

# --- Existing Classes (as provided in previous turns, with update_curriculum modified for batch) ---
class EnvironmentInstance:
    """
    Represents a specific environment configuration identified by a random seed,
    along with its associated Elo rating and performance statistics.
    """
    def __init__(self, seed: int, module_seeds: RNGCollection, initial_elo: int = 1000):
        self.seed = seed
        self.module_seeds = module_seeds
        self.elo = initial_elo
        self.plays = 0
        self.wins = 0 # Agent wins against this environment instance
        self.losses = 0 # Agent losses against this environment instance
        self.status = True # Flag indicating if seed creats working environment

class EloCurriculumManager:
    """
    Manages an Elo-based curriculum for a reinforcement learning agent.
    It tracks agent and environment Elo ratings and selects environments
    strategically based on the agent's current skill level.
    """

    def __init__(self,
                 initial_agent_elo: int = 1000,
                 k_factor_agent: int = 32,
                 k_factor_env: int = 16,
                 target_elo_difference: int = 50, # Target environment Elo is agent_elo + this difference
                 elo_range_for_selection: int = 150): # Select environments within agent_elo +/- this range
        """
        Initializes the EloCurriculumManager.

        Args:
            initial_agent_elo (int): The starting Elo rating for the agent.
            k_factor_agent (int): The K-factor for updating the agent's Elo. Higher means faster changes.
            k_factor_env (int): The K-factor for updating environment Elo.
            target_elo_difference (int): The desired Elo difference between the agent and the
                                         selected environment. Positive means slightly harder.
            elo_range_for_selection (int): When selecting an environment, it will look for environments
                                           whose Elo is within agent_elo +/- this range.
        """
        self.agent_elo = initial_agent_elo
        self.k_factor_agent = k_factor_agent
        self.k_factor_env = k_factor_env
        self.target_elo_difference = target_elo_difference
        self.elo_range_for_selection = elo_range_for_selection

        self.environment_pool: Dict[int, EnvironmentInstance] = {}

    def add_environment(self, seed: int, initial_elo: int = 1000) -> None:
        """Adds a new environment instance to the pool."""
        if seed not in self.environment_pool:
            module_seeds = RNGCollection(seed)
            self.environment_pool[seed] = EnvironmentInstance(seed, module_seeds, initial_elo)

    def new_environment_pool(self, new_seeds: List[int], initial_elo_for_new: int = 1000, 
                                keep_top_n: int = 5, mutate_n: int = 5) -> None:
        """
        Replaces the environment pool using a genetic algorithm-inspired approach:
        - Retains the top N environments based on Elo rating.
        - Creates mutated versions of existing environments by replacing one child RNG.
        - Fills the remaining pool with new environments from provided seeds.

        Args:
            new_seeds (List[int]): List of seeds for new environments.
            initial_elo_for_new (int): Initial Elo for new and mutated environments.
            keep_top_n (int): Number of top environments to retain based on Elo.
            mutate_n (int): Number of environments to create by mutating existing ones.

        Returns:
            None
        """
        # Use target pool size from QuicFireConstants
        target_pool_size = QuicFireConstants.NUM_ENVIRONMENTS_PER_POOL

        # Step 1: Retain top N environments with most similar ELO score
        sorted_envs = sorted(self.environment_pool.items(), 
                               key=lambda x: abs(x[1].elo - self.agent_elo))
        retained_envs = {}
        for seed, env_instance in sorted_envs[:min(keep_top_n, len(self.environment_pool))]:
            # Create new EnvironmentInstance with same RNGs but reset stats
            retained_envs[seed] = EnvironmentInstance(
                seed=seed,
                module_seeds=env_instance.module_seeds,
                initial_elo=initial_elo_for_new
            )
        print(f"  Retained {len(retained_envs)} top environments based on Elo.")

        # Step 2: Select environments to mutate (randomly from existing pool)
        mutable_envs = list(self.environment_pool.items())
        random.shuffle(mutable_envs)
        mutable_envs = mutable_envs[:min(mutate_n, len(mutable_envs))]
        
        # Step 3: Create mutated environments
        mutated_envs = {}
        mutation_seed_counter = max(new_seeds, default=0) + 1 if new_seeds else 100000
        valid_modules = self.environment_pool[0].module_seeds.get_module_names()
        
        for seed, env_instance in mutable_envs:
            new_module_seeds = RNGCollection(
                master_seed=seed,
                initial_seeds={
                    'terrain': env_instance.module_seeds.terrain,
                    'vegetation': env_instance.module_seeds.vegetation,
                    'wind': env_instance.module_seeds.wind,
                    'burn_box': env_instance.module_seeds.burn_box,
                    'ignition': env_instance.module_seeds.ignition
                }
            )
            module_to_mutate = random.choice(valid_modules)
            new_module_seeds.replace_module_seed(module_key=module_to_mutate, new_seed=mutation_seed_counter)
            mutated_envs[mutation_seed_counter] = EnvironmentInstance(seed, new_module_seeds, initial_elo_for_new)
            mutation_seed_counter += 1
        print(f"  Created {len(mutated_envs)} mutated environments.")

        # Step 4: Create new environments from provided seeds
        new_envs = {}
        remaining_slots = max(0, target_pool_size - len(retained_envs) - len(mutated_envs))
        for seed in new_seeds[:remaining_slots]:
            module_seeds = RNGCollection(seed)
            new_envs[seed] = EnvironmentInstance(seed, module_seeds, initial_elo_for_new)
        print(f"  Created {len(new_envs)} new environments from provided seeds.")

        # Step 5: Combine all environments into the new pool
        self.environment_pool = {**retained_envs, **mutated_envs, **new_envs}
        print(f"\n--- Environment Pool Replaced ---")
        print(f"  New Pool Size: {len(self.environment_pool)} "
              f"(Retained: {len(retained_envs)}, Mutated: {len(mutated_envs)}, New: {len(new_envs)})")

        # Warn if pool size doesn't match target
        if len(self.environment_pool) != target_pool_size:
            print(f"  Warning: Pool size ({len(self.environment_pool)}) does not match target ({target_pool_size}).")

    @staticmethod
    def _calculate_elo_change(rating_a: float, rating_b: float, outcome_a: float, k_factor: int) -> float:
        """
        Calculates the Elo rating change for player A.
        Args:
            rating_a (float): Current Elo rating of player A.
            rating_b (float): Current Elo rating of player B.
            outcome_a (float): Actual outcome for player A (1 for win, 0 for loss, 0.5 for draw).
            k_factor (int): K-factor for the rating update.
        Returns:
            float: The change in Elo rating for player A.
        """
        if not 0 <= outcome_a <= 1:
            raise ValueError(f"Outcome must be between 0 and 1, got {outcome_a}")
        
        expected_score_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
        return k_factor * (outcome_a - expected_score_a)

    def update_curriculum(self, 
                           played_env_seed: int, 
                           win_rate: float, 
                           num_episodes_played: int,
                           update_agent_elo: bool = True) -> None:
        """
        Updates the agent's and the played environment's Elo ratings based on a batch of episodes,
        then selects the next environment for the agent to train on.

        Args:
            played_env_seed (int): The random seed of the environment that was just played.
            win_rate (float): The agent's win rate (0.0 to 1.0) over the batch of episodes.
            num_episodes_played (int): The number of episodes in the batch.
            update_agent_elo (bool): If True, update agent's ELO. Set to False during env calibration.
        """
        if played_env_seed not in self.environment_pool:
            raise ValueError(f"Played environment seed {played_env_seed} not found in pool.")

        played_env = self.environment_pool[played_env_seed]
        
        # Outcome for Elo calculation
        agent_outcome = win_rate
        env_outcome = 1.0 - win_rate 

        elo_change_agent = 0.0
        if update_agent_elo:
            # Update agent's Elo
            elo_change_agent = self._calculate_elo_change(
                self.agent_elo, played_env.elo, agent_outcome, self.k_factor_agent
            )
            self.agent_elo += elo_change_agent

        # Update environment's Elo
        elo_change_env = self._calculate_elo_change(
            played_env.elo, self.agent_elo, env_outcome, self.k_factor_env
        )
        played_env.elo += elo_change_env

        # Update environment instance statistics
        played_env.plays += num_episodes_played
        played_env.wins += round(win_rate * num_episodes_played)
        played_env.losses += num_episodes_played - played_env.wins

        print(f"\n--- Post-Batch Update (Env Seed: {played_env_seed}, Win Rate: {win_rate:.2f}, Episodes: {num_episodes_played}) ---")
        print(f"  Agent Elo: {self.agent_elo:.2f} (Change: {elo_change_agent:+.2f} {'(fixed)' if not update_agent_elo else ''})")
        print(f"  Environment {played_env_seed} Elo: {played_env.elo:.2f} (Change: {elo_change_env:+.2f})")
        print(f"  Environment {played_env_seed} Stats: Plays={played_env.plays}, Wins={played_env.wins}, Losses={played_env.losses}")

    def _select_next_environment(self) -> int:
        """
        Selects the next environment for the agent to train on.
        Prioritizes environments whose Elo is close to (agent_elo + target_elo_difference).
        If no such environment exists, it falls back to environments within a broader range.
        If still no environments, it picks a random one, or returns -1 if pool is empty.
        """
        if not self.environment_pool:
            return -1 # No environments available

        desired_elo = self.agent_elo + self.target_elo_difference
        candidate_environments = []
        for env_instance in self.environment_pool.values():
            if abs(env_instance.elo - desired_elo) <= self.elo_range_for_selection:
                candidate_environments.append(env_instance)

        if candidate_environments:
            candidate_environments.sort(key=lambda env: abs(env.elo - desired_elo))
            return candidate_environments[0].seed
        else:
            print(f"  No environments found within +/- {self.elo_range_for_selection} of target Elo {desired_elo:.2f}. "
                  f"Selecting from all available environments.")
            
            all_environments = list(self.environment_pool.values())
            if all_environments:
                all_environments.sort(key=lambda env: abs(env.elo - self.agent_elo))
                return all_environments[0].seed
            else:
                return -1 
            
    def _select_random_environment(self) -> int:
        """
        Selects a random environment from the pool for historical replay, with selection
        probability weighted by the logarithm of the number of plays.

        Returns:
            int: The seed of the selected environment, or -1 if the pool is empty.
        """
        if not self.environment_pool:
            print("Environment pool is empty. Cannot select environment.")
            return -1

        env_seeds = list(self.environment_pool.keys())
        weights = [np.log1p(self.environment_pool[seed].plays + 1) for seed in env_seeds]
        selected_seed = random.choices(env_seeds, weights=weights, k=1)[0]
        
        print(f"Selected environment {selected_seed} with {self.environment_pool[selected_seed].plays} plays "
                    f"(weight: {np.log1p(self.environment_pool[selected_seed].plays + 1):.2f})")
        return selected_seed
            
    def get_environment_seeds(self, seed: int):
        return self.environment_pool[seed].module_seeds
    
    def calculate_generalization_metric(self):
        valid_environments = [env for env in self.environment_pool.values() if env.status]
        if not valid_environments:
            return 0.0
        avg_elo = sum(env.elo for env in valid_environments) / len(valid_environments)
        return 1.0 / avg_elo if avg_elo > 0 else 0.0

# --- Helper Functions for Environment and Agent Interaction ---

def create_training_environment(env_config: Dict, module_seeds: RNGCollection, climate: ClimateConditions) -> FirefighterEnvironment:
    """ Create training environment with module random number generators and config """

    try:
        env = FirefighterEnvironment(
            grid_size=(env_config['obs_grid_x'], env_config['obs_grid_y']),
            max_steps=int(env_config['total_time'] / QuicFireConstants.DT),
            num_agents=env_config['num_agents'],
            sim_base_config=env_config,
            module_seeds=module_seeds,
            climate=climate,
            verbose=False
        )
        return env
    except Exception as e:
        print(f"Failed to create FirefighterEnvironment: {e}")
        raise

def run_single_episode(agent: OptionCriticAgent, env: FirefighterEnvironment) -> Tuple[bool, float, bool]:
    """
    Runs a single episode of the firefighter environment with the agent.
    Returns: True if agent wins, False otherwise, the total reward, and if simulation was successful.
    """
    
    state, info_reset = env.reset()
    episode_reward = 0
    done = False
    truncated = False

    # Reset agent's internal option tracking at the start of each episode
    agent.agent_initial_option_states = [None] * env.num_agents
    agent.agent_chosen_options = [None] * env.num_agents
    agent.agent_accumulated_rewards = [0.0] * env.num_agents
    agent.agent_option_steps = [0] * env.num_agents
    agent.agent_initial_values = [0.0] * env.num_agents
    agent.agent_initial_option_type_log_probs = [0.0] * env.num_agents
    agent.agent_initial_target_x_log_probs = [0.0] * env.num_agents
    agent.agent_initial_target_y_log_probs = [0.0] * env.num_agents
    agent_option_active = [False] * env.num_agents 

    while not done and not truncated:
        actions_for_env_step = [] 

        for agent_idx in range(env.num_agents):
            if env.option_terminated[agent_idx] or not agent_option_active[agent_idx]:
                chosen_option = agent.select_option(state, agent_idx)
                actions_for_env_step.append(
                    [chosen_option['agent_idx'], chosen_option['action_type'], chosen_option['x'], chosen_option['y']]
                )
                agent_option_active[agent_idx] = True
            else:
                current_intent = env.agent_target_intent[agent_idx]
                actions_for_env_step.append(
                    [current_intent['agent_idx'], current_intent['type'], current_intent['x'], current_intent['y']]
                    if current_intent else
                    [agent_idx, env.action_type_map['firebreak'], env.agent_pos[agent_idx][0], env.agent_pos[agent_idx][1]]
                )
        
        actions_for_env_step_np = np.array(actions_for_env_step, dtype=int)
        next_state, per_agent_rewards, done, truncated, info = env.step(actions_for_env_step_np)

        if done and info.get('status') == 'failed':
            return False, 0.0, False # Critical failure is a non-win

        global_reward_this_step = info.get('global_reward', 0.0)
        if 'global_reward' not in info:
            print("Warning: 'global_reward' missing in step info for environment")

        for agent_idx in range(env.num_agents):
            if agent_option_active[agent_idx]:
                blended_reward = (agent.reward_alpha * per_agent_rewards[agent_idx]) + \
                                 ((1 - agent.reward_alpha) * global_reward_this_step)
                
                agent.agent_accumulated_rewards[agent_idx] += blended_reward
                agent.agent_option_steps[agent_idx] += 1

                if env.option_terminated[agent_idx]:
                    final_state_tensor = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0)
                    _, _, _, final_termination_logit, _ = agent.policy_net(final_state_tensor)
                    final_termination_prob = torch.sigmoid(final_termination_logit).item()

                    agent.buffer.store_option_trajectory(
                        initial_state=agent.agent_initial_option_states[agent_idx],
                        chosen_option=agent.agent_chosen_options[agent_idx],
                        accumulated_reward=agent.agent_accumulated_rewards[agent_idx],
                        steps_taken=agent.agent_option_steps[agent_idx],
                        final_state=next_state,
                        initial_value=agent.agent_initial_values[agent_idx],
                        initial_option_type_log_prob=agent.agent_initial_option_type_log_probs[agent_idx],
                        initial_target_x_log_prob=agent.agent_initial_target_x_log_probs[agent_idx],
                        initial_target_y_log_prob=agent.agent_initial_target_y_log_probs[agent_idx],
                        final_termination_prob=final_termination_prob
                    )
                    agent_option_active[agent_idx] = False
                    agent.agent_initial_option_states[agent_idx] = None
                    agent.agent_chosen_options[agent_idx] = None
                    agent.agent_accumulated_rewards[agent_idx] = 0.0
                    agent.agent_option_steps[agent_idx] = 0
                    agent.agent_initial_values[agent_idx] = 0.0
                    agent.agent_initial_option_type_log_probs[agent_idx] = 0.0
                    agent.agent_initial_target_y_log_probs[agent_idx] = 0.0
                    agent.agent_initial_target_x_log_probs[agent_idx] = 0.0 

        state = next_state
        episode_reward += global_reward_this_step 

    is_win = env._check_win_condition() 
    env.close() 
    return is_win, episode_reward, True

def generate_new_environment_seeds(num_environments: int, start_seed: int) -> List[int]:
    """Generates a list of new, unique environment seeds."""
    return [start_seed + i for i in range(num_environments)]

def replace_bad_seed(elo_manager, bad_seed, pool_generation_seed_counter) -> int:
    elo_manager.environment_pool.pop(bad_seed, None)  # Remove bad seed
    new_seed = pool_generation_seed_counter
    elo_manager.add_environment(new_seed, initial_elo=elo_manager.agent_elo)
    pool_generation_seed_counter += 1
    return pool_generation_seed_counter

# =====================================================================
# Main Training Orchestrator Function
def train_firefighter_agent_with_curriculum(base_env_config: Dict, climate: ClimateConditions, tuning: Optional[Dict] = None) -> None:
    """
    Main training loop for firefighter RL agent using Option-Critic with Elo-based curriculum.
    """
    # Apply before create_training_environment divides total_time by DT.
    if tuning:
        apply_tuning(tuning)

    # --- QuicFireConstants for the overall curriculum ---
    BURN_IN_EPISODES = QuicFireConstants.BURN_IN_EPISODES
    BURN_IN_BATCH_SIZE = QuicFireConstants.BURN_IN_BATCH_SIZE
    NUM_ENVIRONMENTS_PER_POOL = QuicFireConstants.NUM_ENVIRONMENTS_PER_POOL
    N_CALIBRATION_EPISODES = QuicFireConstants.N_CALIBRATION_EPISODES
    N_EPISODES_PER_ELO_UPDATE = QuicFireConstants.N_EPISODES_PER_ELO_UPDATE
    N_REGENERATION_PERIOD_BATCHES = QuicFireConstants.N_REGENERATION_PERIOD_BATCHES
    RANDOM_REPLAY_PROB = QuicFireConstants.RANDOM_REPLAY_PROB
    TOTAL_CURRICULUM_BATCHES = QuicFireConstants.TOTAL_CURRICULUM_BATCHES
    CHECKPOINT_INTERVAL = QuicFireConstants.CHECKPOINT_INTERVAL
    ELO_INITIAL_AGENT = QuicFireConstants.ELO_INITIAL_AGENT
    ELO_K_FACTOR_AGENT = QuicFireConstants.ELO_K_FACTOR_AGENT
    ELO_K_FACTOR_ENV = QuicFireConstants.ELO_K_FACTOR_ENV
    ELO_TARGET_DIFFERENCE = QuicFireConstants.ELO_TARGET_DIFFERENCE
    ELO_RANGE_SELECTION = QuicFireConstants.ELO_RANGE_SELECTION
    REWARD_ALPHA = QuicFireConstants.DEFAULT_REWARD_ALPHA

    # Initialize agent
    agent = OptionCriticAgent(env_template=None, reward_alpha=REWARD_ALPHA) 

    # Initialize Elo Manager.
    elo_manager = EloCurriculumManager(
        initial_agent_elo=ELO_INITIAL_AGENT,
        k_factor_agent=ELO_K_FACTOR_AGENT,
        k_factor_env=ELO_K_FACTOR_ENV,
        target_elo_difference=ELO_TARGET_DIFFERENCE,
        elo_range_for_selection=ELO_RANGE_SELECTION
    )

    current_batches_in_regeneration_period = 0 
    pool_generation_seed_counter = 100000 

    print("--- Starting Full Curriculum Training Simulation ---")

    # Save Configuration
    save_config_to_file(base_env_config, "/CurriculumTraining/", "firefighter_policy_config")

    # =====================================================================
    ## Phase 1: Burn-In (Initial Skill Acquisition) 🔥
    print("\n\n===== Phase 1: Burn-In =====")
    burn_in_env_seed = pool_generation_seed_counter 
    burn_in_env_elo = 1000 

    # Add burn-in environment to manager just for its metadata (Elo not updated for it after this phase)
    print(f"Training agent on fixed burn-in environment (Seed: {burn_in_env_seed}, Initial Elo: {burn_in_env_elo}) for {BURN_IN_EPISODES} episodes...")

    # Create Firefighter environment
    burn_in_env_config = base_env_config.copy()
    module_seeds = RNGCollection(pool_generation_seed_counter)
    env = create_training_environment(burn_in_env_config, module_seeds, climate)
    
    total_burn_in_wins = 0
    total_burn_in_episodes = 0

    for i in range(BURN_IN_EPISODES // BURN_IN_BATCH_SIZE):
        batch_wins = 0
        batch_rewards = 0.0
        for _ in range(BURN_IN_BATCH_SIZE):
            is_win, episode_reward, status = run_single_episode(agent, env)

            if not status:
                # Generate a new environment and skip update.
                pool_generation_seed_counter += 1
                module_seeds = RNGCollection(pool_generation_seed_counter)
                env = create_training_environment(burn_in_env_config, module_seeds, climate)
                break

            if is_win:
                batch_wins += 1
            batch_rewards += episode_reward
            agent.update()

        if status:
            total_burn_in_wins += batch_wins
            total_burn_in_episodes += BURN_IN_BATCH_SIZE
            
            win_rate = batch_wins / BURN_IN_BATCH_SIZE
            avg_reward = batch_rewards / BURN_IN_BATCH_SIZE

            # Don't update agent Elo only log so we can still print performance.
            print(f"  Burn-in Batch {i+1} of {BURN_IN_EPISODES // BURN_IN_BATCH_SIZE}: Win Rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f}")

    env.close()

    # =====================================================================
    # Main Training Loop (Orchestrating Phase 2 & 3)
    for total_curriculum_batch_idx in range(TOTAL_CURRICULUM_BATCHES):
        # =====================================================================
        ## Phase 2: Periodic Pool Regeneration/Re-establishment 🌐🔄
        if total_curriculum_batch_idx == 0 or current_batches_in_regeneration_period >= N_REGENERATION_PERIOD_BATCHES or len(elo_manager.environment_pool.keys()) < NUM_ENVIRONMENTS_PER_POOL:
            print(f"\n\n===== Phase 2: Pool Regeneration (Curriculum Batch {total_curriculum_batch_idx+1}) =====")
            
            # 2.1. Generate New Environment Pool
            total_test_seeds = NUM_ENVIRONMENTS_PER_POOL * 2 # Create more than target amount
            new_seeds = generate_new_environment_seeds(total_test_seeds, pool_generation_seed_counter)
            
            # Initialize new environments
            elo_manager.new_environment_pool(
                new_seeds=new_seeds,
                initial_elo_for_new=elo_manager.agent_elo, # Set Environment ELO and Agent ELO to be same
                keep_top_n=QuicFireConstants.KEEP_TOP_N,
                mutate_n=QuicFireConstants.MUTATE_N
            )
            pool_generation_seed_counter += total_test_seeds

            # 2.2. Mini-Burn-In for New Environment Calibration
            print(f"  Starting Mini-Burn-In for {len(new_seeds)} new environments ({N_CALIBRATION_EPISODES} episodes each)")
            working_seeds = []
            for seed in new_seeds:

                # Create Firefighter environment
                env_calibration_config = base_env_config.copy()
                module_seeds = elo_manager.get_environment_seeds(seed)
                env = create_training_environment(env_calibration_config, module_seeds, climate)

                batch_wins = 0
                for _ in range(N_CALIBRATION_EPISODES):
                    is_win, _, status = run_single_episode(agent, env)

                    if not status:
                        # Set flag and skip
                        elo_manager.environment_pool[seed].status = False
                        env.close()
                        break

                    if is_win:
                        batch_wins += 1
                
                if status:
                    win_rate = batch_wins / N_CALIBRATION_EPISODES
                    elo_manager.update_curriculum(
                        played_env_seed=seed,
                        win_rate=win_rate,
                        num_episodes_played=N_CALIBRATION_EPISODES,
                        update_agent_elo=False # Crucial: ONLY update environment's ELO here
                    )

                    working_seeds.append(seed)
                    env.close()

                # Count number of good seeds in environment
                if len(working_seeds) >= NUM_ENVIRONMENTS_PER_POOL:
                    # Remove all seeds either not working or not tested
                    for seed in list(elo_manager.environment_pool.keys()):
                        if seed not in working_seeds:
                            del elo_manager.environment_pool[seed]

            # Track Generalization Measure
            generalization_score = elo_manager.calculate_generalization_metric()
            print(f"  Generalization Score: {generalization_score:.4f} (lower environment Elo indicates better generalization)")

            print(f"  Mini-Burn-In complete. Agent Elo remains: {elo_manager.agent_elo:.2f}")
            current_batches_in_regeneration_period = 0


        if len(elo_manager.environment_pool.keys()) < NUM_ENVIRONMENTS_PER_POOL:
            # Skip training and loop back to generate sufficient pool size.
            continue

        # =====================================================================
        ## Phase 3: Dynamic Curriculum Learning (Main Loop) 🎯🌳
        print(f"\n===== Phase 3: Dynamic Curriculum Learning Batch {total_curriculum_batch_idx+1} of {TOTAL_CURRICULUM_BATCHES} =====")
        current_batches_in_regeneration_period += 1 

        selected_env_seed = -1
        if random.random() < RANDOM_REPLAY_PROB and len(elo_manager.environment_pool) > 0:
            print("  Random Replay triggered!")
            selected_env_seed = elo_manager._select_random_environment()
        else:
            selected_env_seed = elo_manager._select_next_environment()
        
        # 3.2. Batch Episode Execution
        batch_wins = 0
        batch_rewards = 0.0

        # Create Firefighter environment
        env_batch_config = base_env_config.copy()
        module_seeds = elo_manager.get_environment_seeds(selected_env_seed)
        env = create_training_environment(env_batch_config, module_seeds, climate)

        for episode_in_batch in range(N_EPISODES_PER_ELO_UPDATE):
            is_win, episode_reward, status = run_single_episode(agent, env)

            if not status:
                print("ERROR: Main training loop contains bad seed.")
                continue

            if is_win:
                batch_wins += 1
            batch_rewards += episode_reward
            agent.update() # Agent's policy is updated here during active learning

        win_rate = batch_wins / N_EPISODES_PER_ELO_UPDATE
        avg_reward = batch_rewards / N_EPISODES_PER_ELO_UPDATE
        print(f"  Batch completed (Win Rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f} over {N_EPISODES_PER_ELO_UPDATE} episodes)")

        # 3.3. Elo Update (Agent and Environment)
        elo_manager.update_curriculum(
            played_env_seed=selected_env_seed,
            win_rate=win_rate,
            num_episodes_played=N_EPISODES_PER_ELO_UPDATE,
            update_agent_elo=True # Always update agent's ELO in the main learning phase
        )
        env.close()

        # Save checkpoint
        if total_curriculum_batch_idx % CHECKPOINT_INTERVAL == 0 or total_curriculum_batch_idx == TOTAL_CURRICULUM_BATCHES - 1:
            save_checkpoint(base_env_config, agent, "/CurriculumTraining/", "firefighter_policy", total_curriculum_batch_idx)
    
    print("\n--- Full Curriculum Training Simulation Finished ---")

# --- Run the curriculum training ---
if __name__ == "__main__":

    # Define base configuration parameters for the environment
    base_env_config = {
        'nx': 50, # QuicFire simulation x dimension
        'ny': 50, # QuicFire simulation y dimension
        'nz': 10, # QuicFire simulation z dimension
        'dx': 2.0,
        'dy': 2.0,
        'total_time': 10000, # Total simulation time in seconds for an episode
        'obs_grid_x': 20, # Observation grid size X
        'obs_grid_y': 20, # Observation grid size Y
        'num_agents': 2 # Number of agents
    }

    # Define climate conditions
    climate = ClimateConditions(
        temperature=260,  # Kelvin
        humidity=0.6,
        precipitation=600.0  # mm/year
    )

    train_firefighter_agent_with_curriculum(base_env_config=base_env_config, climate=climate)