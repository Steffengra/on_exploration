
from numpy import (
    newaxis,
    argmax,
)
from tensorflow import (
    GradientTape,
    function,
    one_hot,
    reduce_sum,
    reduce_max,
    exp,
)
from tensorflow.keras.models import (
    load_model,
)
from pathlib import (
    Path,
)

from exploration_project_imports.neural_networks import DQNAleatoric


class DQNAleatoricWrap:
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

        # TODO: This should be in config
        self.log_prob_loss_scale = 1e-2

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
        self.dqn.save(Path(model_path, 'dqn_aleatoric'))

    def load_network(
            self,
            model_path: Path,
    ):
        self.dqn = load_model(model_path)

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
            _,
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
            # reduce_sum(next_actions_softmaxes * next_return_estimates)  # appx next state return
        )

        # CALCULATE LOSSES----------------------------------------------------------------------------------------------
        mask = one_hot(action_id, self.num_actions)
        with GradientTape() as tape:
            # TD LOSS
            current_return_estimates, current_action_log_prob_densities = self.dqn.get_action_and_log_prob_density(state)
            current_return_estimates = current_return_estimates[0]
            current_reward_estimate = reduce_sum(current_return_estimates * mask)

            td_error = target_reward_estimate - current_reward_estimate
            loss = td_error ** 2

            # VARIANCE LOSS
            #   std lower -> log prob greater -> reward for keeping log prob lower
            log_prob_density_loss = reduce_sum(current_action_log_prob_densities)
            loss = loss + self.log_prob_loss_scale * log_prob_density_loss


        # CALCULATE GRAD AND APPLY
        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))

        # UPDATE TARGET NETWORK-----------------------------------------------------------------------------------------
        self.update_target_networks(tau_target_update=self.tau_target_network_update)
