
from numpy import (
    ndarray,
    argmax,
    newaxis,
)
from tensorflow import (
    one_hot,
    reduce_sum,
    GradientTape,
)

from exploration_project_imports.neural_networks import DQNDeterministic


class DQNDeterministicWrap:
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

    def train(
            self,
            state,
            action_id,
            reward,
            state_next,
    ) -> None:
        (
            next_action_id,
            next_action_return_estimate,
        ) = self.get_action(state=state_next)

        target_reward_estimate = reward + self.future_reward_discount_gamma * next_action_return_estimate
        mask = one_hot(action_id, self.num_actions)
        with GradientTape() as tape:
            current_return_estimates = self.dqn.call(state[newaxis])
            current_return_estimate = reduce_sum(mask * current_return_estimates)
            td_error = target_reward_estimate - current_return_estimate
            loss = td_error ** 2

        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        # print(gradients)
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))
























