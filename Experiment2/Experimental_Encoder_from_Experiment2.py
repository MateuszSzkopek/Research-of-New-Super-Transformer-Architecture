
# Imports:

import warnings
import os
import random
import time

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D

from sklearn.model_selection import train_test_split
from sklearn import preprocessing as p
from sklearn.metrics import f1_score

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

from Experimental_neurons import NeurogenesisDense

# Classes:

class Positional_Encoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.supports_masking = True
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def compute_mask(self, inputs, mask=None):
        return mask


class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.supports_masking = True
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

    def call(self, x, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, tf.newaxis, :]

        return self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=attention_mask
        )

    def compute_mask(self, inputs, mask=None):
        return mask


class Feed_Forward(tf.keras.layers.Layer):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.neuro_dense = NeurogenesisDense(
            ff_dim, activation="relu", max_units=128
        )
        self.output_dense = Dense(d_model)
        self.ffn = tf.keras.Sequential([
            self.neuro_dense,
            self.output_dense,
        ])
        self._latest_neurogenesis_input = None

    def call(self, x):
        # The callback uses the actual input representation of NeurogenesisDense,
        # rather than the original token ids, to evaluate NORTH-Select.
        object.__setattr__(self, "_latest_neurogenesis_input", x)
        output = self.ffn(x)
        if self.neuro_dense._successor is None:
            # The following Dense layer owns the fan-out weights of new neurons.
            # Linking it lets maybe_grow() zero those weights before activation.
            self.neuro_dense.set_successor(self.output_dense)
        return output


class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, ff_dim):
        super().__init__()
        self.supports_masking = True
        self.attention = Multi_Head_Attention(num_heads=num_heads, d_model=d_model)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = Feed_Forward(d_model=d_model, ff_dim=ff_dim)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask=None):
        attention_output = self.attention(x, mask=mask)
        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)

    def compute_mask(self, inputs, mask=None):
        return mask


class F1_Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x_val, y_val = self.validation_data
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val, verbose=0), axis=1)

        logs["val_f1_macro"] = f1_score(y_true, y_pred, average="macro")
        logs["val_f1_weighted"] = f1_score(y_true, y_pred, average="weighted")


class Neurogenesis_Callback(tf.keras.callbacks.Callback):
    """Collect FFN inputs and apply function-preserving NORTH-Select growth."""
    def __init__(self, feed_forward_layers, grow_every_epochs=1, max_new_neurons=8):
        super().__init__()
        self.feed_forward_layers = feed_forward_layers
        self.grow_every_epochs = grow_every_epochs
        self.max_new_neurons = max_new_neurons

    def on_train_batch_end(self, batch, logs=None):
        for feed_forward in self.feed_forward_layers:
            inputs = feed_forward._latest_neurogenesis_input
            if inputs is not None:
                feed_forward.neuro_dense.observe(inputs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.grow_every_epochs:
            return
        for index, feed_forward in enumerate(self.feed_forward_layers):
            added = feed_forward.neuro_dense.maybe_grow(
                max_new_neurons=self.max_new_neurons
            )
            if logs is not None:
                logs[f"north_added_block_{index}"] = added
                logs[f"north_active_block_{index}"] = int(
                    feed_forward.neuro_dense.active_units.numpy()
                )

# Preprocessing:

NUM_CLASSES = 8
EPOCHS = 1000000000
VOCAB_SIZE = 32768
SEQUENCE_LENGTH = 32
es = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5, restore_best_weights=False)

df = pd.read_csv(r"C:\Users\matis\Desktop\Training_data.csv", sep=",")
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["annotation"], train_size = 0.8, random_state=SEED, shuffle=True)

tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token = "<00V>")
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, maxlen=SEQUENCE_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=SEQUENCE_LENGTH, padding="post")

label_encoder = p.LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Training

config = {
    "vocab_size": VOCAB_SIZE,
    "sequence_length": SEQUENCE_LENGTH,

    "ff_dim": 32,
    "d_model": 16,
    "num_heads": 1,
    "num_blocs": 1,
    
    "num_classes": NUM_CLASSES
    }

layers = [
    Embedding(input_dim=config["vocab_size"], output_dim=config["d_model"], input_length=config["sequence_length"], mask_zero=True),
    Positional_Encoding(max_len=config["sequence_length"], d_model=config["d_model"])
]
for i in range(config["num_blocs"]):
    layers.append(Transformer_Block(num_heads=config["num_heads"], d_model=config["d_model"], ff_dim=config["ff_dim"]))
layers.extend([
    tf.keras.layers.GlobalAveragePooling1D(),
    Dense(config["num_classes"], activation="softmax")
])

start = time.perf_counter()

model = Sequential(layers)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=True,
)
f1_callback = F1_Callback(validation_data=(X_test, y_test))
neurogenesis_callback = Neurogenesis_Callback(
    [layer.ffn for layer in model.layers if isinstance(layer, Transformer_Block)]
)
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=1, callbacks = [es, f1_callback, neurogenesis_callback], shuffle=True)

df = pd.DataFrame(history.history)
end = time.perf_counter()
training_time = end - start
df["training_time_s"] = training_time

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
df.to_csv(os.path.join(desktop, "History.csv"), index=False, encoding="utf-8")

print("\nProgram ended succesfull.\n")



