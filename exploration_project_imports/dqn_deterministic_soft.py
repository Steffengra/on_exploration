
from numpy import (
    argmax,
    newaxis,
    log,
    zeros,
)
from tensorflow import (
    one_hot,
    reduce_sum,
    GradientTape,
    exp,
)
from pathlib import (
    Path,
)

from exploration_project_imports.neural_networks import DQNDeterministic


class DQNDeterministicSoftWrap:
    def __init__(
            self,
            rng,
            hidden_layer_args: dict,
            optimizer,
            optimizer_args: dict,
            future_reward_discount_gamma: float,
    ) -> None:

        self.rng = rng
        self.future_reward_discount_gamma: float = future_reward_discount_gamma

        self.num_actions = hidden_layer_args['num_actions']
        # TODO: Move to config
        self.entropy_scale_alpha: float = 1e-3

        self.dqn = DQNDeterministic(**hidden_layer_args)
        self.dqn.compile(optimizer=optimizer(**optimizer_args))

    def get_action(
            self,
            state,
    ) -> tuple:
        return_estimates = self.dqn.call(state[newaxis]).numpy().flatten()
        action_id = argmax(return_estimates)

        return (
            action_id,
            return_estimates[action_id]
        )

    def save_networks(
            self,
            sample_input,
            model_path: Path,
    ) -> None:
        self.dqn(sample_input[newaxis])  # initialize
        self.dqn.save(Path(model_path, 'dqn_aleatic'))

    def train(
            self,
            state,
            action_id,
            reward,
            state_next,
    ) -> None:
        next_actions_return_estimates = self.dqn.call(state_next[newaxis]).numpy().flatten()
        next_action_id = argmax(next_actions_return_estimates)

        # calculate next state entropy
        # this promotes actions that lead to states with high entropy == similar reward estimates
        next_actions_return_estimates_exp_sum = sum(exp(next_actions_return_estimates))
        next_actions_softmaxes = zeros(self.num_actions)
        for possible_action_id in range(self.num_actions):
            next_actions_softmaxes[possible_action_id] = (
                    exp(next_actions_return_estimates[possible_action_id]) / next_actions_return_estimates_exp_sum
            )

        # print(next_actions_return_estimates[next_action_id])
        # print(self.entropy_scale_alpha * sum(log(next_actions_softmaxes)), '\n')
        target_reward_estimate = reward + self.future_reward_discount_gamma * (
                next_actions_return_estimates[next_action_id]
                - self.entropy_scale_alpha * sum(log(next_actions_softmaxes))
        )
        # print(exp(next_actions_return_estimates[next_action_id]) / sum(exp(next_actions_return_estimates)))
        mask = one_hot(action_id, self.num_actions)
        with GradientTape() as tape:
            current_return_estimates = self.dqn.call(state[newaxis])
            current_return_estimate = reduce_sum(mask * current_return_estimates)
            current_return_estimate_softmax = exp(current_return_estimate) / reduce_sum(exp(current_return_estimates))
            # print(current_return_estimate_softmax)
            # print(target_reward_estimate)
            # print(self.entropy_scale_alpha * current_return_estimate_softmax)
            # print(current_return_estimate)
            td_error = (
                    target_reward_estimate
                    # + self.entropy_scale_alpha * log(current_return_estimate_softmax)
                    - current_return_estimate
            )
            loss = td_error ** 2

        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        # print(gradients)
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))
