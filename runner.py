
from numpy import (
    ones,
    infty,
    newaxis,
    argmax,
)
from tensorflow import (
    convert_to_tensor,
    float32,
)
from datetime import (
    datetime,
)
from shutil import (
    copy2,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    dump as pickle_dump,
)
from pathlib import (
    Path,
)

from exploration_project_imports.simulation import PuncturingSimulation
from exploration_project_imports.dqn_deterministic import DQNDeterministicWrap
from exploration_project_imports.dqn_aleatoric import DQNAleatoricWrap
from exploration_project_imports.dqn_aleatoric_soft import DQNAleatoricSoftWrap


class Runner:
    def __init__(
            self,
            config,
    ) -> None:
        self.config = config

    def train_dqn_aleatoric(
            self,
    ) -> None:
        def progress_print() -> None:
            if (step_id % self.config.steps_per_progress_print) == 0 and (step_id != 0):
                progress = (episode_id + step_id / self.config.num_steps_per_episode) / self.config.num_episodes
                timedelta = datetime.now() - real_time_start
                finish_time = real_time_start + timedelta / progress

                print(f'\rSimulation completed: {progress:.2%}, '
                      f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}',
                      end='')

        def status_print() -> None:
            print(f'\rLast Episode Rewards per Step: '
                  f'{rewards_per_episode[episode_id] / self.config.num_steps_per_episode:.2f}')

        training_name = 'train_dqn_aleatoric'

        sim = PuncturingSimulation(**self.config.sim_args)
        new_state = sim.get_state()
        dqn_aleatoric = DQNAleatoricWrap(**self.config.dqn_args, dummy_input=new_state)

        stats_per_episode = []
        rewards_per_episode = -infty * ones(self.config.num_episodes)

        real_time_start = datetime.now()
        for episode_id in range(self.config.num_episodes):
            sim = PuncturingSimulation(**self.config.sim_args)
            new_state = sim.get_state()
            episode_reward_sum = 0
            for step_id in range(self.config.num_steps_per_episode):
                current_state = new_state.copy()
                action_id, _, _ = dqn_aleatoric.get_action(current_state)
                reward = sim.step(puncture_resource_block_id=action_id)
                new_state = sim.get_state()

                dqn_aleatoric.train(convert_to_tensor(current_state, dtype=float32),
                                    action_id,
                                    convert_to_tensor(reward, dtype=float32),
                                    convert_to_tensor(new_state, dtype=float32))

                episode_reward_sum += reward

                if self.config.verbosity == 1:
                    progress_print()

            stats_per_episode.append(sim.get_stats())
            rewards_per_episode[episode_id] = episode_reward_sum

            if self.config.verbosity == 1:
                status_print()
                print(sim.get_stats())

        for stat in stats_per_episode:
            print(stat)

        # SAVE RESULTS
        metrics = [rewards_per_episode, stats_per_episode]
        with gzip_open(Path(self.config.log_path, f'{training_name}_metrics.gzip'), 'wb') as file:
            pickle_dump(metrics, file=file)

        dqn_aleatoric.save_networks(model_path=self.config.model_path)
        copy2('a_config.py', self.config.model_path)  # save config

    def train_dqn_aleatoric_soft(
            self,
    ) -> None:
        def progress_print() -> None:
            if (step_id % self.config.steps_per_progress_print) == 0 and (step_id != 0):
                progress = (episode_id + step_id / self.config.num_steps_per_episode) / self.config.num_episodes
                timedelta = datetime.now() - real_time_start
                finish_time = real_time_start + timedelta / progress

                print(f'\rSimulation completed: {progress:.2%}, '
                      f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}',
                      end='')

        def status_print() -> None:
            print(f'\rLast Episode Rewards per Step: '
                  f'{rewards_per_episode[episode_id] / self.config.num_steps_per_episode:.2f}')

        training_name = 'train_dqn_aleatoric_soft'

        sim = PuncturingSimulation(**self.config.sim_args)
        new_state = sim.get_state()
        dqn_aleatoric_soft = DQNAleatoricSoftWrap(**self.config.dqn_args, dummy_input=new_state)

        stats_per_episode = []
        rewards_per_episode = -infty * ones(self.config.num_episodes)

        real_time_start = datetime.now()
        for episode_id in range(self.config.num_episodes):
            sim = PuncturingSimulation(**self.config.sim_args)
            new_state = sim.get_state()
            episode_reward_sum = 0
            for step_id in range(self.config.num_steps_per_episode):
                current_state = new_state.copy()
                action_id, _, _ = dqn_aleatoric_soft.get_action(current_state)
                reward = sim.step(puncture_resource_block_id=action_id)
                new_state = sim.get_state()

                dqn_aleatoric_soft.train(convert_to_tensor(current_state, dtype=float32),
                                         action_id,
                                         convert_to_tensor(reward, dtype=float32),
                                         convert_to_tensor(new_state, dtype=float32))

                episode_reward_sum += reward

                if self.config.verbosity == 1:
                    progress_print()

            stats_per_episode.append(sim.get_stats())
            rewards_per_episode[episode_id] = episode_reward_sum

            if self.config.verbosity == 1:
                status_print()
                print(sim.get_stats())

        for stat in stats_per_episode:
            print(stat)

        # SAVE RESULTS
        metrics = [rewards_per_episode, stats_per_episode]
        with gzip_open(Path(self.config.log_path, f'{training_name}_metrics.gzip'), 'wb') as file:
            pickle_dump(metrics, file=file)
        dqn_aleatoric_soft.save_networks(model_path=self.config.model_path)
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

        training_name = 'train_dqn_deterministic'

        exploration_epsilon = self.config.exploration_epsilon_initial
        sim = PuncturingSimulation(**self.config.sim_args)
        new_state = sim.get_state()
        dqn_deterministic = DQNDeterministicWrap(**self.config.dqn_args, dummy_input=new_state)

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

                dqn_deterministic.train(convert_to_tensor(current_state, dtype=float32),
                                        action_id,
                                        convert_to_tensor(reward, dtype=float32),
                                        convert_to_tensor(new_state, dtype=float32))

                episode_reward_sum += reward

                exploration_epsilon = max(0, exploration_epsilon - self.config.exploration_epsilon_decay_per_step)

                if self.config.verbosity == 1:
                    progress_print()

            stats_per_episode.append(sim.get_stats())
            rewards_per_episode[episode_id] = episode_reward_sum

            if self.config.verbosity == 1:
                status_print()
                print(sim.get_stats())

        for stat in stats_per_episode:
            print(stat)

        # SAVE RESULTS
        metrics = [rewards_per_episode, stats_per_episode]
        with gzip_open(Path(self.config.log_path, f'{training_name}_metrics.gzip'), 'wb') as file:
            pickle_dump(metrics, file=file)
        dqn_deterministic.save_networks(model_path=self.config.model_path)
        copy2('a_config.py', self.config.model_path)  # save config

    def train_try_teach_critical(
            self,
    ) -> None:

        num_steps_until_try = {
            'deterministic': 0,
            'aleatoric': 0,
            'aleatoric_soft': 0,
        }

        repeats = 10
        num_steps_until_try_deterministic_total = -infty * ones(repeats)
        num_steps_until_try_aleatoric_total = -infty * ones(repeats)
        num_steps_until_try_aleatoric_soft_total = -infty * ones(repeats)

        for repeat_id in range(repeats):
            sim_deterministic = PuncturingSimulation(**self.config.sim_args)
            sim_aleatoric = PuncturingSimulation(**self.config.sim_args)
            sim_aleatoric_soft = PuncturingSimulation(**self.config.sim_args)
            # sims = [sim_deterministic, sim_aleatoric, sim_aleatoric_soft]
            dummy_state = sim_deterministic.get_state()

            dqn_deterministic = DQNDeterministicWrap(**self.config.dqn_args, dummy_input=dummy_state)
            dqn_aleatoric = DQNAleatoricWrap(**self.config.dqn_args, dummy_input=dummy_state)
            dqn_aleatoric_soft = DQNAleatoricSoftWrap(**self.config.dqn_args, dummy_input=dummy_state)
            # dqns = [dqn_deterministic, dqn_aleatoric, dqn_aleatoric_soft]

            dqn_deterministic.load_network(Path(self.config.model_path, 'dqn_deterministic'))
            dqn_aleatoric.load_network(Path(self.config.model_path, 'dqn_aleatoric'))
            dqn_aleatoric_soft.load_network(Path(self.config.model_path, 'dqn_aleatoric_soft'))

            action_id = 0
            sim = sim_deterministic
            num_steps_until_try_deterministic = 0
            while action_id == 0:
                sim.set_step_id(0)
                sim.set_new_power_gains()
                sim.set_occupation({channel_id: self.config.sim_args['minimum_occupation_length']
                                    for channel_id in range(1, self.config.num_channels + 1)})
                sim.set_puncture_job(critical=True)
                state = sim.get_state()

                return_estimates = dqn_deterministic.dqn.call(state[newaxis])
                action_id = argmax(return_estimates)

                reward = sim.step(action_id)
                new_state = sim.get_state()
                dqn_deterministic.train(convert_to_tensor(state, dtype=float32),
                                        action_id,
                                        convert_to_tensor(reward, dtype=float32),
                                        convert_to_tensor(new_state, dtype=float32))

                num_steps_until_try_deterministic += 1
                # if num_steps_until_try_deterministic % 10 == 0:
                #     print('d', dqn_deterministic.dqn.call(state[newaxis]))
                if num_steps_until_try_deterministic % 10_000 == 0:
                    print('d', 'not achieved within 10_000 steps')
                    break

            num_steps_until_try['deterministic'] = num_steps_until_try_deterministic
            num_steps_until_try_deterministic_total[repeat_id] = num_steps_until_try_deterministic

            action_id = 0
            sim = sim_aleatoric
            num_steps_until_try_aleatoric = 0
            while action_id == 0:
                sim.set_step_id(0)
                sim.set_new_power_gains()
                sim.set_occupation({channel_id: self.config.sim_args['minimum_occupation_length']
                                    for channel_id in range(1, self.config.num_channels + 1)})
                sim.set_puncture_job(critical=True)
                state = sim.get_state()

                return_estimates, _ = dqn_aleatoric.dqn.get_action_and_log_prob_density(state[newaxis])
                action_id = argmax(return_estimates)

                reward = sim.step(action_id)
                new_state = sim.get_state()
                dqn_aleatoric.train(convert_to_tensor(state, dtype=float32),
                                    action_id,
                                    convert_to_tensor(reward, dtype=float32),
                                    convert_to_tensor(new_state, dtype=float32))

                num_steps_until_try_aleatoric += 1
                # if num_steps_until_try_aleatoric % 10 == 0:
                #     print('a', dqn_aleatoric.dqn.call(state[newaxis]))
                if num_steps_until_try_aleatoric % 10_000 == 0:
                    print('a', 'not achieved within 10_000 steps')
                    break

            num_steps_until_try['aleatoric'] = num_steps_until_try_aleatoric
            num_steps_until_try_aleatoric_total[repeat_id] = num_steps_until_try_aleatoric

            action_id = 0
            sim = sim_aleatoric_soft
            num_steps_until_try_aleatoric_soft = 0
            while action_id == 0:
                sim.set_step_id(0)
                sim.set_new_power_gains()
                sim.set_occupation({channel_id: self.config.sim_args['minimum_occupation_length']
                                    for channel_id in range(1, self.config.num_channels + 1)})
                sim.set_puncture_job(critical=True)
                state = sim.get_state()

                return_estimates, _ = dqn_aleatoric_soft.dqn.get_action_and_log_prob_density(state[newaxis])
                action_id = argmax(return_estimates)

                reward = sim.step(action_id)
                new_state = sim.get_state()
                dqn_aleatoric_soft.train(convert_to_tensor(state, dtype=float32),
                                         action_id,
                                         convert_to_tensor(reward, dtype=float32),
                                         convert_to_tensor(new_state, dtype=float32))

                num_steps_until_try_aleatoric_soft += 1
                # if num_steps_until_try_aleatoric_soft % 10 == 0:
                #     print('s', dqn_aleatoric_soft.dqn.call(state[newaxis]))
                if num_steps_until_try_aleatoric_soft % 10_000 == 0:
                    print('s', 'not achieved within 10_000 steps')
                    break

            num_steps_until_try['aleatoric_soft'] = num_steps_until_try_aleatoric_soft
            num_steps_until_try_aleatoric_soft_total[repeat_id] = num_steps_until_try_aleatoric_soft

            print(num_steps_until_try)

        print(num_steps_until_try_deterministic_total)
        print(num_steps_until_try_aleatoric_total)
        print(num_steps_until_try_aleatoric_soft_total)

        metrics = [num_steps_until_try_deterministic_total,
                   num_steps_until_try_aleatoric_total,
                   num_steps_until_try_aleatoric_soft_total]
        with gzip_open(Path(self.config.log_path, f'training_try_critical_metrics.gzip'), 'wb') as file:
            pickle_dump(metrics, file=file)

    def test_critical(
            self,
    ) -> None:
        sim_deterministic = PuncturingSimulation(**self.config.sim_args)
        sim_aleatoric = PuncturingSimulation(**self.config.sim_args)
        sim_aleatoric_soft = PuncturingSimulation(**self.config.sim_args)
        sims = [sim_deterministic, sim_aleatoric, sim_aleatoric_soft]
        dummy_state = sim_deterministic.get_state()

        dqn_deterministic = DQNDeterministicWrap(**self.config.dqn_args, dummy_input=dummy_state)
        dqn_aleatoric = DQNAleatoricWrap(**self.config.dqn_args, dummy_input=dummy_state)
        dqn_aleatoric_soft = DQNAleatoricSoftWrap(**self.config.dqn_args, dummy_input=dummy_state)
        dqns = [dqn_deterministic, dqn_aleatoric, dqn_aleatoric_soft]

        dqn_deterministic.load_network(Path(self.config.model_path, 'dqn_deterministic'))
        dqn_aleatoric.load_network(Path(self.config.model_path, 'dqn_aleatoric'))
        dqn_aleatoric_soft.load_network(Path(self.config.model_path, 'dqn_aleatoric_soft'))

        repeats = 1
        magnitude_difference_deterministic = -infty * ones(repeats)
        magnitude_difference_aleatoric = -infty * ones(repeats)
        magnitude_difference_aleatoric_soft = -infty * ones(repeats)

        logstd_action_1_aleatoric = -infty * ones(repeats)
        mean_logstd_action_23_aleatoric = -infty * ones(repeats)
        logstd_action_1_aleatoric_soft = -infty * ones(repeats)
        mean_logstd_action_23_aleatoric_soft = -infty * ones(repeats)

        for repeat_id in range(repeats):
            for sim in sims:
                sim.set_step_id(0)
                sim.set_new_power_gains()
                sim.set_occupation({channel_id: self.config.sim_args['minimum_occupation_length']
                                    for channel_id in range(1, self.config.num_channels + 1)})
                sim.set_puncture_job(critical=True)

            state = sim_deterministic.get_state()
            print(sim_deterministic.rb_power_gains)
            action = dqn_deterministic.dqn.call(state[newaxis])
            action = action.numpy().flatten()
            magnitude_difference_deterministic[repeat_id] = action[0] / ((action[1] + action[2]) / 2)

            state = sim_aleatoric.get_state()
            action, log_stds = dqn_aleatoric.dqn.call(state[newaxis])
            action = action.numpy().flatten()
            log_stds = log_stds.numpy().flatten()
            magnitude_difference_aleatoric[repeat_id] = action[0] / ((action[1] + action[2]) / 2)
            logstd_action_1_aleatoric[repeat_id] = log_stds[0]
            mean_logstd_action_23_aleatoric[repeat_id] = (log_stds[1] + log_stds[2]) / 2

            state = sim_aleatoric_soft.get_state()
            action, log_stds = dqn_aleatoric_soft.dqn.call(state[newaxis])
            action = action.numpy().flatten()
            log_stds = log_stds.numpy().flatten()
            magnitude_difference_aleatoric_soft[repeat_id] = action[0] / ((action[1] + action[2]) / 2)
            logstd_action_1_aleatoric_soft[repeat_id] = log_stds[0]
            mean_logstd_action_23_aleatoric_soft[repeat_id] = (log_stds[1] + log_stds[2]) / 2

        metrics = [
            [magnitude_difference_deterministic, magnitude_difference_aleatoric, magnitude_difference_aleatoric_soft],
            [logstd_action_1_aleatoric, logstd_action_1_aleatoric_soft],
            [mean_logstd_action_23_aleatoric, mean_logstd_action_23_aleatoric_soft]
        ]
        with gzip_open(Path(self.config.log_path, f'testing_critical_metrics.gzip'), 'wb') as file:
            pickle_dump(metrics, file=file)
