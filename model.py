import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np
from tensorflow.keras.backend import print_tensor
from tensorflow.keras.optimizers import Adam

def _build_sensor_model_2d(input_shape):
    """
    Builds a sub-model for processing sensor data.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(8, (3, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(16, (3, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.2)(x)
    
    batch_size, height, width, channels = x.shape.as_list()
    new_shape = (height, width * channels)
    
    x = layers.Reshape(new_shape)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(16, activation='tanh', return_sequences=True)(x)
    x = layers.GlobalAveragePooling1D()(x)

    return Model(inputs, x)

def build_sensor_model_2d_small(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(8, (3, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(16, (3, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    batch_size, height, width, channels = x.shape.as_list()
    new_shape = (height, width * channels)
    
    x = layers.Reshape(new_shape)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(16, activation='tanh', return_sequences=True)(x)
    x = layers.GlobalAveragePooling1D()(x)

    return Model(inputs, x)

# def build_sensor_model_1d(input_shape):
#     """
#     Builds a sub-model for processing 1D sensor data.

#     Args:
#     input_shape (tuple): Shape of the input data for the sensor.

#     Returns:
#     tf.keras.Model: The sub-model for the sensor.
#     """
#     inputs = Input(shape=input_shape)
#     x = layers.Conv1D(32, 3, activation='relu')(inputs)
#     x = layers.MaxPooling1D(2)(x)
#     x = layers.Conv1D(64, 3, activation='relu')(x)
#     x = layers.MaxPooling1D(2)(x)
#     x = layers.GRU(64, activation='relu', return_sequences=True)(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     return Model(inputs, x)

def _build_combined_model(input_shapes, num_classes,model_config):
    """
    Builds a classification model that combines data from three sensors.

    Args:
    input_shape (tuple): Shape of the input data for each sensor.
    num_classes (int): Number of output classes.

    Returns:
    tf.keras.Model: The compiled classification model.
    """
    # Create sub-models for each sensor
    sensor1_model = build_sensor_model_2d(input_shapes[0])
    sensor2_model = build_sensor_model_2d(input_shapes[1])
    sensor3_model = build_sensor_model_2d(input_shapes[2])
    #sensor4_model = build_sensor_model_2d_small(input_shapes[3])
    
    sensor1_model.summary()
    sensor2_model.summary()
    sensor3_model.summary()
    #sensor4_model.summary()
    
    # Define inputs for each sensor
    sensor1_input = Input(shape=input_shapes[0])
    sensor2_input = Input(shape=input_shapes[1])
    sensor3_input = Input(shape=input_shapes[2])
    #sensor4_input = Input(shape=input_shapes[3])

    # Get the outputs from each sub-model
    sensor1_output = sensor1_model(sensor1_input)
    sensor2_output = sensor2_model(sensor2_input)
    sensor3_output = sensor3_model(sensor3_input)
    #sensor4_output = sensor4_model(sensor4_input)

    # Concatenate the outputs
    concatenated = layers.concatenate([sensor1_output, sensor2_output, sensor3_output])

    # Add Dense and Dropout layers
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    optimizer = Adam(learning_rate=model_config['lr'])
    
    # Create and compile the model
    model = Model(inputs=[sensor1_input, sensor2_input, sensor3_input], outputs=output)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',  # Use 'categorical_crossentropy' if labels are one-hot encoded
                  #loss='mean_squared_error',
                  #metrics=['mae']
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=0.5),
                           tf.keras.metrics.Recall(thresholds=0.5)]
                 )

                  #metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparsePrecision(), tf.keras.metrics.SparseRecall()])

    return model

def build_sensor_model_2d(input_shape, model_config):
    """
    Builds a sub-model for processing sensor data.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1'], model_config['kernel_size_1'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    x = layers.Conv2D(model_config['num_filters_2'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    batch_size, height, width, channels = x.shape.as_list()
    new_shape = (height, width * channels)
    
    x = layers.Reshape(new_shape)(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    x = layers.LSTM(model_config['lstm_units'], activation='tanh', return_sequences=True)(x)
    x = layers.GlobalAveragePooling1D()(x)

    return Model(inputs, x)

def build_combined_model(input_shapes, num_classes, model_config):
    """
    Builds a classification model that combines data from three sensors.

    Args:
    input_shapes (list of tuples): Shapes of the input data for each sensor.
    num_classes (int): Number of output classes.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The compiled classification model.
    """
    # Create sub-models for each sensor
    sensor1_model = build_sensor_model_2d(input_shapes[0], model_config)
    sensor2_model = build_sensor_model_2d(input_shapes[1], model_config)
    sensor3_model = build_sensor_model_2d(input_shapes[2], model_config)
    sensor4_model = build_sensor_model_2d(input_shapes[2], model_config)
    
    # Define inputs for each sensor
    sensor1_input = Input(shape=input_shapes[0])
    sensor2_input = Input(shape=input_shapes[1])
    sensor3_input = Input(shape=input_shapes[2])
    sensor4_input = Input(shape=input_shapes[3])

    # Get the outputs from each sub-model
    sensor1_output = sensor1_model(sensor1_input)
    sensor2_output = sensor2_model(sensor2_input)
    sensor3_output = sensor3_model(sensor3_input)
    sensor4_output = sensor3_model(sensor4_input)

    # Concatenate the outputs
    concatenated = layers.concatenate([sensor1_output, sensor2_output, sensor3_output,sensor4_output])

    # Add Dense and Dropout layers
    x = layers.Dense(model_config['dense_units'], activation='relu')(concatenated)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    optimizer = Adam(learning_rate=model_config['lr'])
    
    # Create and compile the model
    model = Model(inputs=[sensor1_input, sensor2_input, sensor3_input,sensor4_input], outputs=output)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=0.5),
                           tf.keras.metrics.Recall(thresholds=0.5)]
                 )

    return model
# def build_combined_model_2(input_shapes, num_classes):
#     """
#     Builds a classification model that combines data from three sensors.

#     Args:
#     input_shape (tuple): Shape of the input data for each sensor.
#     num_classes (int): Number of output classes.

#     Returns:
#     tf.keras.Model: The compiled classification model.
#     """
#     # Create sub-models for each sensor
#     sensor1_model = build_sensor_model_2d(input_shapes[0])
#     sensor2_model = build_sensor_model_2d(input_shapes[1])
#     sensor3_model = build_sensor_model_2d_small(input_shapes[2])
    
#     sensor1_model.summary()
#     sensor2_model.summary()
#     sensor3_model.summary()
    
#     # Define inputs for each sensor
#     sensor1_input = Input(shape=input_shapes[0])
#     sensor2_input = Input(shape=input_shapes[1])
#     sensor3_input = Input(shape=input_shapes[2])

#     # Get the outputs from each sub-model
#     sensor1_output = sensor1_model(sensor1_input)
#     sensor2_output = sensor2_model(sensor2_input)
#     sensor3_output = sensor3_model(sensor3_input)

#     # Concatenate the outputs
#     concatenated = layers.concatenate([sensor1_output, sensor2_output, sensor3_output])

#     # Add Dense and Dropout layers
#     x = layers.Dense(64, activation='relu')(concatenated)
#     x = layers.Dropout(0.5)(x)
#     output = layers.Dense(1, activation='sigmoid')(x)

#     # Create and compile the model
#     model = Model(inputs=[sensor1_input, sensor2_input, sensor3_input], outputs=output)
    
#     optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    
#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',  # Use 'categorical_crossentropy' if labels are one-hot encoded
#                   #loss='mean_squared_error',
#                   #metrics=['mae']
#                   metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=0.5),
#                            tf.keras.metrics.Recall(thresholds=0.5)]
#                  )

#                   #metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparsePrecision(), tf.keras.metrics.SparseRecall()])

#     return model

if __name__ == '__main__':
    # Define input shape and number of classes
    ppg_input_shape = (750, 3, 1)  # Adjust as per your data
    gyro_input_shape = (750, 3, 1)  # Adjust as per your data
    acc_input_shape = (750, 3, 1)  # Adjust as per your data
    temp_input_shape = (6, 2, 1)

    input_shapes = [ppg_input_shape, gyro_input_shape, acc_input_shape, temp_input_shape]

    num_classes = 2  # Adjust as per your number of classes

    # Build and summarize the model
    model = build_combined_model(input_shapes, num_classes)
    model.summary()

    # Create synthetic test input data
    ppg_data = np.random.rand(1, *ppg_input_shape).astype(np.float32)  # Batch size of 1
    gyro_data = np.random.rand(1, *gyro_input_shape).astype(np.float32)
    acc_data = np.random.rand(1, *acc_input_shape).astype(np.float32)
    temp_data = np.random.rand(1, *temp_input_shape).astype(np.float32)

    # Print shapes of the synthetic data
    print("PPG data shape:", ppg_data.shape)
    print("Gyro data shape:", gyro_data.shape)
    print("Acc data shape:", acc_data.shape)
    print("Temp data shape:", temp_data.shape)

    # Pass the synthetic data through the model to get the output
    output = model.predict([ppg_data, gyro_data, acc_data, temp_data])

    # Print the model output
    print("Model output:", output)