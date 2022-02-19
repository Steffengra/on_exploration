
from numpy.random import (
    default_rng,
)
from pathlib import (
    Path,
)

from exploration_project_imports.dl_internals_with_expl import (
    optimizer_adam,
)


class Config:
    def __init__(
            self,
    ) -> None:
        self.simulation_title: str = 'test'
        self.verbosity: int = 1  # 0=off, 1=yes
        self.steps_per_progress_print: int = 100

        self.num_episodes: int = 30
        self.num_steps_per_episode: int = 3_000

        self.num_channels: int = 2

        self.sim_args: dict = {
            'num_steps_per_frame': 7,
            'minimum_occupation_length': 5,
            'probability_rb_occupation': 0.7,
            'probability_new_puncture_request': 0.1,
            'probability_critical_request': 0.01,
        }
        self.dqn_args: dict = {
            'future_reward_discount_gamma': 0.99,
            'hidden_layer_args': {
                'hidden_layer_units': [128, 128],
                'activation_hidden': 'relu',  # [>'relu', 'elu', 'tanh' 'penalized_tanh']
                'kernel_initializer_hidden': 'glorot_normal',  # options: tf.keras.initializers, default: >'glorot_uniform',
            },
            'optimizer': optimizer_adam,
            'optimizer_args': {
                'learning_rate': 1e-4,
                'amsgrad': True,
            },
        }
        self.exploration_epsilon_initial: float = 0.99  # for deterministic only
        decay_exploration_epsilon_to_zero_at: float = 0.5  # for deterministic only

        self.reward_weights: dict = {
            'sum_capacity': 1.0,
            'puncture_miss': 5.0,
            'critical_puncture_miss': 5.0,
            'immediate_puncture': 0.0,
        }

        # INTERNAL------------------------------------------------------------------------------------------------------
        self.rng = default_rng()
        self.dqn_args['rng'] = self.rng
        self.sim_args['rng'] = self.rng

        self.sim_args['num_resource_blocks'] = self.num_channels
        self.dqn_args['hidden_layer_args']['num_actions'] = self.num_channels + 1
        self.sim_args['reward_weights'] = self.reward_weights
        self.sim_args['verbosity'] = self.verbosity

        self.model_path = Path(Path.cwd(), 'zz_saved_models', self.simulation_title)
        self.log_path = Path(Path.cwd(), 'zz_logs', self.simulation_title)

        self.exploration_epsilon_decay_per_step = (
                self.exploration_epsilon_initial /
                (decay_exploration_epsilon_to_zero_at * self.num_steps_per_episode * self.num_episodes)
        )

        # POST-INIT-----------------------------------------------------------------------------------------------------
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
