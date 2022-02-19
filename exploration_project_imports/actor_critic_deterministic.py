
from exploration_project_imports.neural_networks import (
    ValueNetwork,
    PolicyNetworkDeterministic,
)


# TODO: Add target nets?

class ActorCriticDeterministic:
    def __init__(
            self,
            rng,
            hidden_layer_args_critic: dict,
            hidden_layer_args_actor: dict,
            optimizer_critic,
            optimizer_critic_args: dict,
            optimizer_actor,
            optimizer_actor_args: dict,
            future_reward_discount_gamma: float,
    ) -> None:
        self.rng = rng
        self.future_reward_discount_gamma = future_reward_discount_gamma

        # INITIALIZE NETWORKS-------------------------------------------------------------------------------------------
        self.critic = ValueNetwork(**hidden_layer_args_critic)
        self.actor = PolicyNetworkDeterministic(**hidden_layer_args_actor)

        self.critic.compile(optimizer=optimizer_critic(**optimizer_critic_args))
        self.actor.compile(optimizer=optimizer_actor(**optimizer_actor_args))

    def train(
            self,
            state,
            action,
            reward,
            next_state,
    ) -> None:
        next_action = self.actor.call(next_state)
        q_estimate = self.critic.call([next_state, next_action])




























