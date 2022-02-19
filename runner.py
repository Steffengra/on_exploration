
from numpy import (
    ones,
    infty,
)
from datetime import (
    datetime,
)
from shutil import (
    copy2,
)

from exploration_project_imports.simulation import PuncturingSimulation
from exploration_project_imports.dqn_deterministic import DQNDeterministicWrap
from exploration_project_imports.dqn_aleatic import DQNAleaticWrap


class Runner:
    def __init__(
            self,
            config,
    ) -> None:
        self.config = config

    def train_dqn_aleatic(
            self,
    ) -> None:
        def progress_print() -> None:
            if (step_id % self.config.steps_per_progress_print) == 0 and (step_id != 0):
                progress = (episode_id + step_id / self.config.num_steps_per_episode) / self.config.num_episodes
                timedelta = datetime.now() - real_time_start
                finish_time = real_time_start + timedelta / progress

                print(f'\rSimulation completed: {progress:.2%}, '
                      f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

        def status_print() -> None:
            print(f'\rLast Episode Rewards per Step: {rewards_per_episode[episode_id] / self.config.num_steps_per_episode:.2f}')

        dqn_aleatic = DQNAleaticWrap(**self.config.dqn_args)

        stats_per_episode = []
        rewards_per_episode = -infty * ones(self.config.num_episodes)

        real_time_start = datetime.now()
        for episode_id in range(self.config.num_episodes):
            sim = PuncturingSimulation(**self.config.sim_args)
            new_state = sim.get_state()
            episode_reward_sum = 0
            for step_id in range(self.config.num_steps_per_episode):
                current_state = new_state.copy()
                action_id, _, _ = dqn_aleatic.get_action(current_state)
                reward = sim.step(puncture_resource_block_id=action_id)
                new_state = sim.get_state()

                dqn_aleatic.train(current_state, action_id, reward, new_state)

                episode_reward_sum += reward

                if self.config.verbosity == 1:
                    progress_print()

            stats_per_episode.append(sim.get_stats())
            rewards_per_episode[episode_id] = episode_reward_sum

            if self.config.verbosity == 1:
                status_print()

        for stat in stats_per_episode:
            print(stat)

        dqn_aleatic.save_networks(model_path=self.config.model_path, sample_input=new_state)
        copy2('a_config.py', self.config.model_path)  # save config

    def train_dqn_deterministic(
            self,
    ) -> None:
        def progress_print() -> None:
            if (step_id % self.config.steps_per_progress_print) == 0 and (step_id != 0):
                progress = (episode_id + step_id / self.config.num_steps_per_episode) / self.config.num_episodes
                timedelta = datetime.now() - real_time_start
                finish_time = real_time_start + timedelta / progress

                print(f'\rSimulation completed: {progress:.2%}, '
                      f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}', end='')

        def status_print() -> None:
            print(f'\rLast Episode Rewards per Step: '
                  f'{rewards_per_episode[episode_id] / self.config.num_steps_per_episode:.2f}, '
                  f'Current Exploration Epsilon: '
                  f'{exploration_epsilon:.2f}')

        def epsilon_greedy_exploration(determined_action_id) -> int:
            if self.config.rng.random() < exploration_epsilon:
                return self.config.rng.choice(self.config.num_channels + 1)
            else:
                return determined_action_id

        exploration_epsilon = self.config.exploration_epsilon_initial
        dqn_deterministic = DQNDeterministicWrap(**self.config.dqn_args)

        stats_per_episode = []
        rewards_per_episode = -infty * ones(self.config.num_episodes)

        real_time_start = datetime.now()
        for episode_id in range(self.config.num_episodes):
            sim = PuncturingSimulation(**self.config.sim_args)
            new_state = sim.get_state()
            episode_reward_sum = 0
            for step_id in range(self.config.num_steps_per_episode):
                current_state = new_state.copy()
                action_id, _ = dqn_deterministic.get_action(current_state)
                action_id = epsilon_greedy_exploration(action_id)
                reward = sim.step(puncture_resource_block_id=action_id)
                new_state = sim.get_state()

                dqn_deterministic.train(current_state, action_id, reward, new_state)

                episode_reward_sum += reward

                exploration_epsilon = max(0, exploration_epsilon - self.config.exploration_epsilon_decay_per_step)

                if self.config.verbosity == 1:
                    progress_print()

            stats_per_episode.append(sim.get_stats())
            rewards_per_episode[episode_id] = episode_reward_sum

            if self.config.verbosity == 1:
                status_print()

        for stat in stats_per_episode:
            print(stat)

        dqn_deterministic.save_networks(model_path=self.config.model_path, sample_input=new_state)
        copy2('a_config.py', self.config.model_path)  # save config
