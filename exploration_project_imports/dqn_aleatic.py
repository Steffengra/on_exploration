
from numpy import (
    newaxis,
    argmax,
)
from tensorflow import (
    GradientTape,
)

from exploration_project_imports.neural_networks import DQNAleatic


class DQNAleaticWrap:
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

        self.dqn = DQNAleatic(**hidden_layer_args)
        self.dqn.compile(optimizer=optimizer(**optimizer_args))

    def get_action(
            self,
            state,
    ) -> tuple:
        (
            reward_estimates,
            reward_estimate_log_probs,
        ) = self.dqn.get_action_and_log_prob_density(state[newaxis])
        action = argmax(reward_estimates)

        return action, reward_estimates.numpy().flatten()[action], reward_estimate_log_probs.numpy().flatten()[action]

    def train(
            self,
            state,
            action_id,
            reward,
            state_next,
    ) -> None:
        (
            next_action_id,
            next_reward_estimate,
            next_reward_estimate_log_probs,
        ) = self.get_action(state_next)

        target_reward_estimate = reward + self.future_reward_discount_gamma * next_reward_estimate
        with GradientTape() as tape:
            current_reward_estimates = self.dqn.call(state[newaxis])
            td_error = target_reward_estimate - current_reward_estimates[action_id]
            loss = td_error ** 2

        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        print(gradients)
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))
