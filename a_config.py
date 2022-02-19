
from numpy.random import (
    default_rng,
)

from exploration_project_imports.dl_internals_with_expl import (
    optimizer_adam,
)


class Config:
    def __init__(
            self,
    ) -> None:
        self.simulation_name: str = 'test'
        self.verbosity: int = 1  # 0=off, 1=yes
        self.steps_per_progress_print: int = 100

        self.num_episodes: int = 10
        self.num_steps_per_episode: int = 10_000

        self.num_channels: int = 2

        self.sim_args: dict = {
            'num_steps_per_frame': 7,
            'minimum_occupation_length': 5,
            'probability_rb_occupation': 0.7,
            'probability_new_puncture_request': 0.1,
            'probability_critical_request': 0.01,
        }
        self.dqn_args: dict = {
            'future_reward_discount_gamma': 0.9,
            'hidden_layer_args': {
                'hidden_layer_units': [2, 2],
                'activation_hidden': 'relu',  # [>'relu', 'elu', 'tanh' 'penalized_tanh']
                'kernel_initializer_hidden': 'glorot_normal',  # options: tf.keras.initializers, default: >'glorot_uniform',
            },
            'optimizer': optimizer_adam,
            'optimizer_args': {
                'learning_rate': 1e-3,
                'epsilon': 1e-8,
                'amsgrad': True,
            },
        }
        self.reward_weights: dict = {
            'sum_capacity': 1.0,
            'puncture_miss': 1.0,
            'critical_puncture_miss': 1.0,
            'immediate_puncture': 1.0,
        }

        # INTERNAL------------------------------------------------------------------------------------------------------
        rng = default_rng(seed=1)
        self.dqn_args['rng'] = rng
        self.sim_args['rng'] = rng

        self.sim_args['num_resource_blocks'] = self.num_channels
        self.dqn_args['hidden_layer_args']['num_actions'] = self.num_channels + 1
        self.sim_args['reward_weights'] = self.reward_weights
        self.sim_args['verbosity'] = self.verbosity
