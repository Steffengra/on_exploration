
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
    reduce_max,
    GradientTape,
    clip_by_value,
    function,
)
from tensorflow.math import (
    log as tf_log,
)
from tensorflow.keras.optimizers import(
    SGD,
)
from pathlib import (
    Path,
)

from exploration_project_imports.neural_networks import DQNAleatoric


class DQNAleatoricSoftWrap:
    def __init__(
            self,
            rng,
            hidden_layer_args: dict,
            optimizer,
            optimizer_args: dict,
            future_reward_discount_gamma: float,
            tau_target_network_update: float,
            dummy_input,
    ) -> None:
        self.rng = rng
        self.future_reward_discount_gamma: float = future_reward_discount_gamma
        self.tau_target_network_update: float = tau_target_network_update

        # TODO: Maybe move to config
        # Gradients are applied on the log value. This way, entropy_scale_alpha is restricted to positive range
        self.log_entropy_scale_alpha = tf_Variable(log(1), trainable=True, dtype=float32)
        self.target_entropy: float = 1.0
        self.entropy_scale_alpha_optimizer = SGD(learning_rate=1e-6)

        self.num_actions = hidden_layer_args['num_actions']

        self.dqn = DQNAleatoric(**hidden_layer_args)
        self.dqn_target = DQNAleatoric(**hidden_layer_args)
        self.dqn.compile(optimizer=optimizer(**optimizer_args))
        self.dqn(dummy_input[newaxis])  # initialize weights
        self.dqn_target(dummy_input[newaxis])  # initialize weights
        self.update_target_networks(tau_target_update=1.0)

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

    def update_target_networks(
            self,
            tau_target_update: float
    ) -> None:
        for v_primary, v_target in zip(self.dqn.trainable_variables,
                                       self.dqn_target.trainable_variables):
            v_target.assign(tau_target_update * v_primary + (1 - tau_target_update) * v_target)

    def save_networks(
            self,
            model_path: Path,
    ) -> None:
        self.dqn.save(Path(model_path, 'dqn_aleatoric_soft'))

    @function
    def train(
            self,
            state,
            action_id,
            reward,
            state_next,
    ) -> None:
        # CONSTRUCT TARGET RETURN ESTIMATE------------------------------------------------------------------------------
        (
            next_return_estimates,
            next_reward_estimate_log_probs,
        ) = self.dqn_target.get_action_and_log_prob_density(state_next[newaxis])

        next_return_estimates = next_return_estimates[0]

        # CALCULATE NEXT STATE ENTROPY
        #   this promotes actions that lead to states with high entropy == similar reward estimates
        # next_actions_return_estimates_exp_sum = reduce_sum(exp(next_return_estimates))
        # next_actions_softmaxes = [
        #     exp(next_return_estimates[possible_action_id]) / next_actions_return_estimates_exp_sum
        #     for possible_action_id in range(self.num_actions)]

        # ASSEMBLE TARGET RETURN ESTIMATE
        target_reward_estimate = reward + self.future_reward_discount_gamma * (
            reduce_max(next_return_estimates)  # max next state return
            # sum(next_actions_softmaxes * next_return_estimates)  # appx next state return
            # - exp(self.log_entropy_scale_alpha) * sum(log(next_actions_softmaxes))  # appx next entropy
        )

        # CALCULATE LOSSES----------------------------------------------------------------------------------------------
        mask = one_hot(action_id, self.num_actions)
        with GradientTape() as tape:
            # TD LOSS
            current_return_estimates, _ = self.dqn.get_action_and_log_prob_density(state)
            current_return_estimates = current_return_estimates[0]
            current_reward_estimate = reduce_sum(current_return_estimates * mask)

            td_error = target_reward_estimate - current_reward_estimate
            loss = td_error ** 2

            # ENTROPY LOSS
            current_actions_return_estimates_exp_sum = reduce_sum(exp(current_return_estimates))
            current_actions_softmaxes = [
                exp(current_return_estimates[possible_action_id]) / current_actions_return_estimates_exp_sum
                for possible_action_id in range(self.num_actions)
            ]

            #   clip for numerical stability
            current_actions_log_softmaxes = tf_log(clip_by_value(current_actions_softmaxes,
                                                                 clip_value_min=1e-3, clip_value_max=1))
            loss = loss - exp(self.log_entropy_scale_alpha) * reduce_sum(current_actions_log_softmaxes)

        # CALCULATE GRAD AND APPLY
        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))

        # ADJUST ENTROPY SCALE ALPHA------------------------------------------------------------------------------------
        # TRAIN ENTROPY SCALE ALPHA TO TARGET
        # with GradientTape() as tape:
        #     alpha_loss = (-exp(self.log_entropy_scale_alpha) * reduce_sum(log(current_actions_softmaxes))
        #                   - self.target_entropy) ** 2
        # alpha_gradients = tape.gradient(target=alpha_loss, sources=[self.log_entropy_scale_alpha])
        # self.entropy_scale_alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_entropy_scale_alpha]))

        # ANNEAL ALPHA
        # with GradientTape() as tape:
        #     alpha_loss = exp(self.log_entropy_scale_alpha)
        # alpha_gradients = tape.gradient(target=alpha_loss, sources=[self.log_entropy_scale_alpha])
        # self.entropy_scale_alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_entropy_scale_alpha]))

        # UPDATE TARGET NETWORK-----------------------------------------------------------------------------------------
        self.update_target_networks(tau_target_update=self.tau_target_network_update)
