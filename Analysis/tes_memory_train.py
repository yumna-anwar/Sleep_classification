import tensorflow as tf
import numpy as np
import gc
from tensorflow.keras import backend as K

# Define a simple model for binary classification
def build_simple_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate random data
def generate_random_data(num_samples, input_shape):
    X = np.random.random((num_samples,) + input_shape).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    return X, y

# Generate random training and validation data
input_shape = (64, 64, 1)  # Example input shape (image-like, grayscale)
X_train, y_train = generate_random_data(10000, input_shape)
X_val, y_val = generate_random_data(2000, input_shape)

# Create TensorFlow datasets for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Build the model
model = build_simple_model(input_shape)
model.summary()

# Define a simple training loop with callbacks
class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()

# Train the model with a callback to collect garbage and clear session
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,  # Set to a small number for quick testing
    callbacks=[GarbageCollectionCallback()],
    verbose=1
)

# Clear session and collect garbage after training
K.clear_session()
gc.collect()

