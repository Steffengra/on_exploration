
from numpy import (
    newaxis,
    argmax,
    log,
)
from tensorflow import (
    float32,
    Variable as tf_Variable,
    exp,
    one_hot,
    reduce_sum,
    GradientTape,
)
from tensorflow.keras.optimizers import(
    SGD,
)
from pathlib import (
    Path,
)

from exploration_project_imports.neural_networks import DQNAleatic


class DQNAleaticSoftWrap:
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

        # TODO: Maybe move to config
        # Gradients are applied on the log value. This way, entropy_scale_alpha is restricted to positive range
        self.log_entropy_scale_alpha = tf_Variable(log(1), trainable=True, dtype=float32)
        self.target_entropy: float = 1.0
        self.entropy_scale_alpha_optimizer = SGD(learning_rate=1e-4)

        self.num_actions = hidden_layer_args['num_actions']

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

    def save_networks(
            self,
            sample_input,
            model_path: Path,
    ) -> None:
        self.dqn(sample_input[newaxis])  # initialize
        self.dqn.save(Path(model_path, 'dqn_aleatic_soft'))

    def train(
            self,
            state,
            action_id,
            reward,
            state_next,
    ) -> None:
        (
            _,
            next_reward_estimate,
            next_return_estimate_log_prob_density,
        ) = self.get_action(state_next)

        # Tune return estimate to prefer actions that lead to states with high variance
        target_reward_estimate = reward + self.future_reward_discount_gamma * (
            next_reward_estimate
            - exp(self.log_entropy_scale_alpha) * next_return_estimate_log_prob_density
        )
        mask = one_hot(action_id, self.num_actions)
        with GradientTape() as tape:
            current_reward_estimates, current_reward_log_prob_densities = self.dqn.get_action_and_log_prob_density(state)
            current_reward_estimate = reduce_sum(current_reward_estimates * mask)
            # TODO: UNDERSTAND THIS
            current_reward_log_prob_density = reduce_sum(current_reward_log_prob_densities * mask)
            current_reward_estimate_augmented = (
                    current_reward_estimate
                    + exp(self.log_entropy_scale_alpha) * current_reward_log_prob_density
            )
            td_error = target_reward_estimate - current_reward_estimate_augmented
            loss = td_error ** 2

        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        # for layer in gradients:
        #     print(layer)
        # print('\n\n')
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))

        # train entropy scale alpha
        with GradientTape() as tape:
            alpha_loss = -exp(self.log_entropy_scale_alpha) * current_reward_log_prob_density + self.target_entropy
        alpha_gradients = tape.gradient(target=alpha_loss, sources=[self.log_entropy_scale_alpha])
        self.entropy_scale_alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_entropy_scale_alpha]))
