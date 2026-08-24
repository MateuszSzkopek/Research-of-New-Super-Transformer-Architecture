# Imports

import tensorflow as tf


# Classes

# class Dense(tf.keras.layers.Dense):
#     It is already implemented in TensorFlow and can be imported directly.


class NeurogenesisDense(tf.keras.layers.Dense):
    """A Dense layer with function-preserving NORTH-Select growth.

    ``units`` is the number of initially active neurons.  ``max_units`` reserves
    the maximum width of the layer, which keeps Keras tensor shapes static while
    allowing individual neurons to be activated during training.

    To preserve the network function when a neuron is added, attach the layer
    that consumes this layer's output with :meth:`set_successor`.  Its fan-out
    weights for every new neuron are then set to zero before activation.
    """

    def __init__(
        self,
        units,
        *,
        max_units,
        growth_threshold=0.97,
        candidate_count=100,
        buffer_size=1024,
        rank_epsilon=1e-2,
        **kwargs,
    ):
        if max_units < units:
            raise ValueError("max_units must be at least units.")
        if not 0.0 < growth_threshold <= 1.0:
            raise ValueError("growth_threshold must be in (0, 1].")
        if candidate_count < 1 or buffer_size < 1:
            raise ValueError("candidate_count and buffer_size must be positive.")

        # Dense's `units` is deliberately the reserved, static output width.
        # `initial_units` is the number of neurons that initially participate.
        super().__init__(units=max_units, **kwargs)
        self.initial_units = int(units)
        self.max_units = int(max_units)
        self.growth_threshold = float(growth_threshold)
        self.candidate_count = int(candidate_count)
        self.buffer_size = int(buffer_size)
        self.rank_epsilon = float(rank_epsilon)
        self._successor = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("NeurogenesisDense needs a known final input dimension.")
        if self.quantization_mode or self.lora_rank:
            raise NotImplementedError(
                "NeurogenesisDense does not support quantized or LoRA Dense layers."
            )
        self._input_dim = int(input_dim)

        # We allocate the maximum capacity once.  Resizing a tf.Variable after
        # a model is built is not supported by Keras or by most optimizers.
        # Keras 3 exposes `kernel` as a read-only property backed by `_kernel`.
        self._kernel = self.add_weight(
            name="kernel",
            shape=(self._input_dim, self.max_units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.max_units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None

        self.active_units = self.add_weight(
            name="active_units",
            shape=(),
            dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(self.initial_units),
            trainable=False,
        )
        self._active_mask = self.add_weight(
            name="active_mask",
            shape=(self.max_units,),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Constant(
                [1.0] * self.initial_units + [0.0] * (self.max_units - self.initial_units)
            ),
            trainable=False,
        )
        self._input_buffer = self.add_weight(
            name="north_input_buffer",
            shape=(self.buffer_size, self._input_dim),
            initializer="zeros",
            dtype=self.dtype,
            trainable=False,
        )
        self._buffer_count = self.add_weight(
            name="north_buffer_count",
            shape=(),
            dtype=tf.int32,
            initializer="zeros",
            trainable=False,
        )
        self._buffer_position = self.add_weight(
            name="north_buffer_position",
            shape=(),
            dtype=tf.int32,
            initializer="zeros",
            trainable=False,
        )
        self._baseline_effective_dimension = self.add_weight(
            name="north_baseline_effective_dimension",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=False,
        )
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=2, axes={-1: self._input_dim}
        )
        self.built = True

    def call(self, inputs):
        outputs = tf.tensordot(inputs, self.kernel, axes=[[-1], [0]])
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Apply this after activation: e.g. sigmoid(0) is not zero.
        outputs = outputs * self._active_mask
        if self.activity_regularizer is not None:
            self.add_loss(self.activity_regularizer(outputs))
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape).concatenate(self.max_units)

    def set_successor(self, layer):
        """Attach the next layer so growth can be function-preserving.

        The successor must already be built and its input dimension must equal
        this layer's ``max_units``.  It may be either Dense or
        NeurogenesisDense.
        """
        if not getattr(layer, "built", False):
            raise ValueError("Build the successor before attaching it.")
        kernel = getattr(layer, "kernel", None)
        if kernel is None or kernel.shape[0] != self.max_units:
            raise ValueError(
                "The successor must consume this layer's static max_units output."
            )
        # It is a runtime reference, not a nested Keras sub-layer.  Bypass
        # Keras' state tracker because this link is established after build.
        object.__setattr__(self, "_successor", layer)
        return layer

    def observe(self, inputs):
        """Add representative inputs to the activation buffer used by NORTH."""
        self._require_eager("observe")
        inputs = tf.cast(tf.convert_to_tensor(inputs), self.compute_dtype)
        inputs = tf.reshape(inputs, (-1, self._input_dim))
        take = min(int(tf.shape(inputs)[0]), self.buffer_size)
        if take == 0:
            return
        inputs = inputs[-take:]
        position = int(self._buffer_position.numpy())
        indices = (tf.range(take, dtype=tf.int32) + position) % self.buffer_size
        self._input_buffer.assign(
            tf.tensor_scatter_nd_update(self._input_buffer, indices[:, tf.newaxis], inputs)
        )
        self._buffer_position.assign((position + take) % self.buffer_size)
        self._buffer_count.assign(min(self.buffer_size, int(self._buffer_count.numpy()) + take))

    def calibrate(self, inputs=None):
        """Set NORTH's reference effective dimension from representative data."""
        self._require_eager("calibrate")
        if inputs is not None:
            self.observe(inputs)
        reference_inputs = self._reference_inputs()
        activations = self._active_activations(reference_inputs)
        self._baseline_effective_dimension.assign(
            self._effective_dimension(activations)
        )
        return float(self._baseline_effective_dimension.numpy())

    def maybe_grow(self, inputs=None, max_new_neurons=None):
        """Run the NORTH-Select trigger and activate selected new neurons.

        Call this after an optimizer step.  ``inputs`` are the inputs *to this
        layer*, not necessarily the model's original inputs.  The method returns
        the number of neurons added.  It runs eagerly by design because Keras
        cannot change layer state/structure safely inside a traced ``fit`` step.
        """
        self._require_eager("maybe_grow")
        if inputs is not None:
            self.observe(inputs)

        active = int(self.active_units.numpy())
        room = self.max_units - active
        if room == 0:
            return 0

        reference_inputs = self._reference_inputs()
        if int(tf.shape(reference_inputs)[0]) <= active:
            # The paper's numerical-rank measure requires more samples than
            # active neurons; wait until the buffer is sufficiently populated.
            return 0

        activations = self._active_activations(reference_inputs)
        effective_dimension = self._effective_dimension(activations)
        if float(self._baseline_effective_dimension.numpy()) < 0.0:
            self._baseline_effective_dimension.assign(effective_dimension)
        baseline = float(self._baseline_effective_dimension.numpy())
        gap = float(effective_dimension.numpy()) - self.growth_threshold * baseline
        if gap <= 0.0:
            return 0

        # Eq. 3 in the paper scales growth by current width.  The minimum of
        # one avoids a permanent stall for small layers (where floor(M * gap)=0).
        count = max(1, int(active * gap))
        count = min(count, room)
        if max_new_neurons is not None:
            count = min(count, int(max_new_neurons))
        if count <= 0:
            return 0

        selected = self._north_select(reference_inputs, activations, count)
        start, stop = active, active + count

        # Setting fan-out connections to zero keeps the network output exactly
        # unchanged at the moment of growth.  Their gradients are non-zero on
        # the next backward pass, so the new neurons can subsequently join in.
        if self._successor is not None:
            successor_kernel = self._successor.kernel
            successor_kernel.assign(
                tf.concat(
                    [
                        successor_kernel[:start],
                        tf.zeros_like(successor_kernel[start:stop]),
                        successor_kernel[stop:],
                    ],
                    axis=0,
                )
            )

        updated_kernel = tf.concat(
            [self.kernel[:, :start], selected, self.kernel[:, stop:]], axis=1
        )
        self.kernel.assign(updated_kernel)
        if self.bias is not None:
            self.bias.assign(
                tf.concat(
                    [self.bias[:start], tf.zeros((count,), self.bias.dtype), self.bias[stop:]],
                    axis=0,
                )
            )
        self._active_mask.assign(
            tf.concat(
                [
                    self._active_mask[:start],
                    tf.ones((count,), self._active_mask.dtype),
                    self._active_mask[stop:],
                ],
                axis=0,
            )
        )
        self.active_units.assign(stop)
        return count

    def _reference_inputs(self):
        count = int(self._buffer_count.numpy())
        if count == 0:
            raise ValueError("No NORTH samples available; pass inputs or call observe first.")
        return self._input_buffer[:count]

    def _active_activations(self, inputs):
        active = int(self.active_units.numpy())
        outputs = tf.linalg.matmul(inputs, self.kernel[:, :active])
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias[:active])
        return self.activation(outputs) if self.activation is not None else outputs

    def _effective_dimension(self, activations):
        activations = tf.cast(activations, tf.float32)
        sample_count = tf.cast(tf.shape(activations)[0], activations.dtype)
        singular_values = tf.linalg.svd(activations / tf.sqrt(sample_count), compute_uv=False)
        rank = tf.reduce_sum(tf.cast(singular_values > self.rank_epsilon, tf.float32))
        return rank / tf.cast(tf.shape(activations)[1], tf.float32)

    def _north_select(self, inputs, existing_activations, count):
        """Select candidates with the most novel post-activations, sequentially."""
        selected_weights = []
        activations = tf.cast(existing_activations, self.compute_dtype)
        active = int(self.active_units.numpy())
        target_norm = tf.reduce_mean(tf.norm(self.kernel[:, :active], axis=0))

        for _ in range(count):
            candidates = self.kernel_initializer(
                shape=(self._input_dim, self.candidate_count), dtype=self.compute_dtype
            )
            candidate_norms = tf.norm(candidates, axis=0, keepdims=True)
            candidates = candidates * target_norm / tf.maximum(candidate_norms, tf.keras.backend.epsilon())
            candidate_activations = tf.linalg.matmul(inputs, candidates)
            if self.activation is not None:
                candidate_activations = self.activation(candidate_activations)

            # Match NORTH-Select's objective: maximize the numerical effective
            # dimension after adding the candidate.  Residual energy provides a
            # deterministic tie-breaker when several candidates increase rank.
            best_index, best_score, best_residual = 0, -float("inf"), -float("inf")
            for index in range(self.candidate_count):
                candidate = candidate_activations[:, index : index + 1]
                score = float(
                    self._effective_dimension(tf.concat([activations, candidate], axis=1)).numpy()
                )
                residual = self._orthogonal_residual(activations, candidate)
                if score > best_score or (score == best_score and residual > best_residual):
                    best_index, best_score, best_residual = index, score, residual

            weight = candidates[:, best_index : best_index + 1]
            selected_weights.append(weight)
            activations = tf.concat(
                [activations, candidate_activations[:, best_index : best_index + 1]], axis=1
            )

        return tf.concat(selected_weights, axis=1)

    def _orthogonal_residual(self, activations, candidate):
        """Fraction of a candidate activation outside the current SVD subspace."""
        matrix = tf.cast(activations, tf.float32)
        singular_values, left_vectors, _ = tf.linalg.svd(matrix, full_matrices=False)
        rank = int(tf.reduce_sum(tf.cast(singular_values > self.rank_epsilon, tf.int32)).numpy())
        candidate = tf.cast(candidate, tf.float32)
        if rank == 0:
            return float(tf.norm(candidate).numpy())
        basis = left_vectors[:, :rank]
        residual = candidate - tf.linalg.matmul(basis, tf.linalg.matmul(basis, candidate, transpose_a=True))
        return float(tf.math.divide_no_nan(tf.norm(residual), tf.norm(candidate)).numpy())

    @staticmethod
    def _require_eager(method):
        if not tf.executing_eagerly():
            raise RuntimeError(
                f"NeurogenesisDense.{method} must run eagerly. Use a custom training loop "
                "or compile(model, run_eagerly=True) for growth steps."
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.initial_units,
                "max_units": self.max_units,
                "growth_threshold": self.growth_threshold,
                "candidate_count": self.candidate_count,
                "buffer_size": self.buffer_size,
                "rank_epsilon": self.rank_epsilon,
            }
        )
        return config
