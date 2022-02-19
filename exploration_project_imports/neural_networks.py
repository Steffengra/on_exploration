
from abc import (
    ABC,
)
from tensorflow import (
    Tensor,
    float32 as tf_float32,
    function as tf_function,
    expand_dims as tf_expand_dims,
    exp as tf_exp,
    clip_by_value as tf_clip_by_value,
    print as tf_print,
)
from tensorflow.keras import (
    Model,
)
from tensorflow.keras.layers import (
    Dense,
)
from tensorflow_probability import (
    distributions as tf_distributions,
)

from exploration_project_imports.activation_functions import (
    activation_penalized_tanh,
)


class DQNDeterministic(Model, ABC):
    def __init__(
            self,
            num_actions: int,
            hidden_layer_units: list,
            activation_hidden: str = 'relu',
            kernel_initializer_hidden: str = 'glorot_uniform',
    ) -> None:
        super().__init__()

        # ACTIVATION----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh

        # LAYERS--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for units in hidden_layer_units:
            self.hidden_layers.append(
                Dense(
                    units=units,
                    kernel_initializer=kernel_initializer_hidden,  # default='glorot_uniform'
                    activation=activation_hidden,  # default=None
                    bias_initializer='zeros'  # default='zeros'
                )
            )
        self.output_layer = Dense(units=num_actions, dtype=tf_float32)

    @tf_function
    def call(
            self,
            inputs,
            training=None,
            masks=None,
    ) -> tuple[Tensor, Tensor]:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return_estimates = self.output_layer(x)

        return return_estimates


class DQNAleatoric(Model, ABC):
    def __init__(
            self,
            num_actions: int,
            hidden_layer_units: list,
            activation_hidden: str = 'relu',
            kernel_initializer_hidden: str = 'glorot_uniform',
    ) -> None:
        super().__init__()

        # ACTIVATION----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh

        # LAYERS--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for units in hidden_layer_units:
            self.hidden_layers.append(
                Dense(
                    units=units,
                    kernel_initializer=kernel_initializer_hidden,  # default='glorot_uniform'
                    activation=activation_hidden,  # default=None
                    bias_initializer='zeros'  # default='zeros'
                )
            )
        self.output_layer_means = Dense(units=num_actions, dtype=tf_float32)
        self.output_layer_log_stds = Dense(units=num_actions, dtype=tf_float32)

    @tf_function
    def call(
            self,
            inputs,
            training=None,
            masks=None,
    ) -> tuple[Tensor, Tensor]:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        means = self.output_layer_means(x)
        log_stds = self.output_layer_log_stds(x)

        # log_stds are typically clipped in implementations. [-20, 2] seems to be the popular interval.
        #  Clipping logs by such a wide range should not have much of an impact.
        log_stds = tf_clip_by_value(log_stds, -20, 2)

        return (
            means,
            log_stds
        )

    @tf_function
    def get_action_and_log_prob_density(
            self,
            state,
    ) -> tuple[Tensor, Tensor]:
        if state.shape.ndims == 1:
            state = tf_expand_dims(state, axis=0)

        means, log_stds = self.call(state)
        stds = tf_exp(log_stds)
        distributions = tf_distributions.Normal(loc=means, scale=stds)
        actions = distributions.sample()
        action_log_prob_densities = distributions.log_prob(actions)

        return (
            actions,
            action_log_prob_densities,
        )
