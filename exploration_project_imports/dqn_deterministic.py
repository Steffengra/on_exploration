
from numpy import (
    newaxis,
    argmax,
)
from tensorflow import (
    one_hot,
    reduce_sum,
    reduce_max,
    GradientTape,
    function,
)
from pathlib import (
    Path,
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
            dummy_input,
    ) -> None:

        self.rng = rng
        self.future_reward_discount_gamma: float = future_reward_discount_gamma

        self.num_actions = hidden_layer_args['num_actions']

        self.dqn = DQNDeterministic(**hidden_layer_args)
        self.dqn_target = DQNDeterministic(**hidden_layer_args)
        self.dqn.compile(optimizer=optimizer(**optimizer_args))
        self.dqn(dummy_input[newaxis])  # initialize weights
        self.dqn_target(dummy_input[newaxis])  # initialize weights
        self.update_target_networks(tau_target_update=1.0)

    def get_action(
            self,
            state,
    ) -> tuple:
        return_estimates = self.dqn_target.call(state[newaxis])[0]
        action_id = argmax(return_estimates)

        return (
            action_id,
            return_estimates[action_id]
        )

    def save_networks(
            self,
            model_path: Path,
    ) -> None:
        self.dqn.save(Path(model_path, 'dqn_aleatoric'))

    def update_target_networks(
            self,
            tau_target_update: float
    ) -> None:
        for v_primary, v_target in zip(self.dqn.trainable_variables,
                                       self.dqn_target.trainable_variables):
            v_target.assign(tau_target_update * v_primary + (1 - tau_target_update) * v_target)

    @function
    def train(
            self,
            state,
            action_id,
            reward,
            state_next,
    ) -> None:
        # CONSTRUCT TARGET RETURN ESTIMATE------------------------------------------------------------------------------
        next_action_return_estimates = self.dqn_target.call(state_next[newaxis])
        target_reward_estimate = reward + self.future_reward_discount_gamma * reduce_max(next_action_return_estimates)

        # CALCULATE LOSSES----------------------------------------------------------------------------------------------
        mask = one_hot(action_id, self.num_actions)
        with GradientTape() as tape:
            current_return_estimates = self.dqn.call(state[newaxis])
            current_return_estimate = reduce_sum(mask * current_return_estimates)
            td_error = target_reward_estimate - current_return_estimate
            loss = td_error ** 2

        # CALCULATE GRAD AND APPLY
        parameters = self.dqn.trainable_variables
        gradients = tape.gradient(target=loss, sources=parameters)
        self.dqn.optimizer.apply_gradients(zip(gradients, parameters))

        # UPDATE TARGET NETWORK-----------------------------------------------------------------------------------------
        # TODO: Move to config
        self.update_target_networks(tau_target_update=0.01)
