# Imports

import tensorflow as tf


# Classes

# class Dense(tf.keras.layers.Dense):
#     It is already implemented in TensorFlow and can be imported directly.


class InfomorphicDense(tf.keras.layers.Dense):
    """Dense layer with receptive and contextual input compartments.

    The layer implements the modulatory activation function used by
    infomorphic neurons for supervised classification::

        A(F, C, L) = F * ((1 - a_c - a_l)
                          + a_c * sigmoid(beta_c * F * C)
                          + a_l * sigmoid(beta_l * F * L))

    ``F`` is derived from the standard feed-forward input, ``C`` from context
    (for example, a one-hot label available during training), and optional
    ``L`` from lateral input. The dimensions of ``C`` and ``L`` do not need to
    equal the number of units because every compartment has its own kernel.

    The input to ``call`` can take one of the following forms:

    * ``layer(x)`` -- inference without contextual input;
    * ``layer((x, context))`` -- bivariate variant;
    * ``layer((x, context, lateral))`` -- when ``trivariate=True``.

    The first layer call must include every compartment that will be trained,
    so that TensorFlow can infer its dimensions. ``context`` and ``lateral``
    may subsequently be omitted; they are then replaced with zeros.
    """

    def __init__(
        self,
        units,
        activation=None,
        *,
        trivariate=False,
        context_strength=0.1,
        lateral_strength=0.1,
        context_beta=2.0,
        lateral_beta=2.0,
        **kwargs,
    ):
        """Creates the layer.

        Args:
            units: Number of output neurons.
            activation: Final Keras activation, e.g. ``"sigmoid"``.
            trivariate: Whether to support a third, lateral compartment.
            context_strength: Strength of contextual modulation (alpha_c).
            lateral_strength: Strength of lateral modulation (alpha_l).
            context_beta: Slope of the F--C interaction.
            lateral_beta: Slope of the F--L interaction.
            **kwargs: Remaining arguments of ``tf.keras.layers.Dense``.
        """
        if not 0.0 <= context_strength <= 1.0:
            raise ValueError("context_strength must be in [0, 1].")
        if not 0.0 <= lateral_strength <= 1.0:
            raise ValueError("lateral_strength must be in [0, 1].")
        if context_strength + lateral_strength > 1.0:
            raise ValueError(
                "context_strength + lateral_strength must not exceed 1."
            )

        super().__init__(units=units, activation=activation, **kwargs)
        self.trivariate = trivariate
        self.context_strength = context_strength
        self.lateral_strength = lateral_strength
        self.context_beta = context_beta
        self.lateral_beta = lateral_beta
        self.context_kernel = None
        self.lateral_kernel = None
        # Dense accepts one tensor by default. This layer accepts a structure
        # (x, context[, lateral]), so input validation is handled locally.
        self.input_spec = None

    @staticmethod
    def _is_multi_input_shape(input_shape):
        return (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        )

    def build(self, input_shape):
        if self._is_multi_input_shape(input_shape):
            feedforward_shape = tf.TensorShape(input_shape[0])
            context_shape = (
                tf.TensorShape(input_shape[1]) if len(input_shape) > 1 else None
            )
            lateral_shape = (
                tf.TensorShape(input_shape[2]) if len(input_shape) > 2 else None
            )
        else:
            feedforward_shape = tf.TensorShape(input_shape)
            context_shape = None
            lateral_shape = None

        # Dense creates the receptive kernel and bias, and handles its usual
        # options, including regularization and weight constraints.
        super().build(feedforward_shape)
        self.input_spec = None

        if context_shape is not None:
            if context_shape[-1] is None:
                raise ValueError("The final context dimension must be known.")
            self.context_kernel = self.add_weight(
                name="context_kernel",
                shape=(int(context_shape[-1]), self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

        if self.trivariate and lateral_shape is not None:
            if lateral_shape[-1] is None:
                raise ValueError("The final lateral dimension must be known.")
            self.lateral_kernel = self.add_weight(
                name="lateral_kernel",
                shape=(int(lateral_shape[-1]), self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

    def _parse_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError("InfomorphicDense requires a feed-forward input.")
            if len(inputs) > 3:
                raise ValueError("Provide at most (x, context, lateral).")
            feedforward = inputs[0]
            context = inputs[1] if len(inputs) > 1 else None
            lateral = inputs[2] if len(inputs) > 2 else None
        else:
            feedforward, context, lateral = inputs, None, None

        if lateral is not None and not self.trivariate:
            raise ValueError("Lateral input requires trivariate=True.")
        return feedforward, context, lateral

    def call(self, inputs):
        feedforward, context, lateral = self._parse_inputs(inputs)

        # F is the standard receptive Dense signal before final activation.
        f = tf.linalg.matmul(feedforward, self.kernel)
        if self.use_bias:
            f = tf.nn.bias_add(f, self.bias)

        if context is not None:
            if self.context_kernel is None:
                raise ValueError(
                    "The layer was built without context. Its first call must "
                    "be layer((x, context))."
                )
            c = tf.linalg.matmul(context, self.context_kernel)
            context_term = tf.math.sigmoid(self.context_beta * f * c)
        else:
            # Missing context means neutral modulation: sigmoid(0) = 0.5.
            context_term = tf.fill(tf.shape(f), tf.cast(0.5, f.dtype))

        if self.trivariate and lateral is not None:
            if self.lateral_kernel is None:
                raise ValueError(
                    "The layer was built without lateral input. Its first call "
                    "must be layer((x, context, lateral))."
                )
            l = tf.linalg.matmul(lateral, self.lateral_kernel)
            lateral_term = tf.math.sigmoid(self.lateral_beta * f * l)
        else:
            lateral_term = tf.fill(tf.shape(f), tf.cast(0.5, f.dtype))

        # Without modulatory signals, the factor equals 1, so the layer behaves
        # like Dense before its final activation.
        modulation = (
            1.0
            - 0.5 * self.context_strength
            - 0.5 * self.lateral_strength
            + self.context_strength * context_term
            + self.lateral_strength * lateral_term
        )
        output = f * modulation
        return self.activation(output) if self.activation is not None else output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "trivariate": self.trivariate,
                "context_strength": self.context_strength,
                "lateral_strength": self.lateral_strength,
                "context_beta": self.context_beta,
                "lateral_beta": self.lateral_beta,
            }
        )
        return config


