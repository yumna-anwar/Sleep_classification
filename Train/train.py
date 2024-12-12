import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from optuna.trial import FixedTrial
from tensorflow.keras.models import load_model
import os
import gc

#FROM PROJECT FILES
from Data_loader import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.run_functions_eagerly(False)

gc.set_threshold(700, 10, 10)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()
        
def count_labels(df, label_col):
    # Count the occurrences of each label (0 and 1)
    label_counts = df[label_col].value_counts()
    num_zeros = label_counts.get(0, 0)  # Get the count of zeros, default to 0 if not present
    num_ones = label_counts.get(1, 0)   # Get the count of ones, default to 0 if not present
    return num_zeros, num_ones

def stratified_batching(dataset, batch_size, output_shapes, output_types):
    def generator():
        data, labels = [], []
        for features, label in dataset:
            data.append(features)
            labels.append(label.numpy())
        data = np.array(data)
        labels = np.array(labels)

        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        min_len = min(len(pos_indices), len(neg_indices))

        pos_indices = np.random.choice(pos_indices, min_len, replace=False)
        neg_indices = np.random.choice(neg_indices, min_len, replace=False)

        indices = np.concatenate([pos_indices, neg_indices])
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = data[batch_indices]
            batch_labels = labels[batch_indices]

            # Convert data and labels to tensors, adding the additional dimension
            batch_data_tensors = tuple([tf.convert_to_tensor(sensor_data).numpy().reshape(batch_size, *sensor_data.shape[1:], 1) for sensor_data in zip(*batch_data)])
            batch_labels_tensor = tf.convert_to_tensor(batch_labels)

            yield batch_data_tensors, batch_labels_tensor

    return tf.data.Dataset.from_generator(
        generator,
        output_types=output_types,
        output_shapes=output_shapes
    )
def Model_objective(trial, config):
    try:
        random_seed = 100
        features_acc = config['features']['accelerometer']
        features_gyro = config['features']['gyroscope']
        features_ppg = config['features']['ppg']
        features_temp = config['features']['temperature']
        label_col = config['labels']['label_column']

        # TUNE WITH OPTUNA

        model_config = {
            'lr': trial.suggest_loguniform('learning_rate', 1e-6, 1e-2),
            'num_filters_1': trial.suggest_categorical('num_filters_1', [8, 16, 32, 64]),
            'num_filters_2': trial.suggest_categorical('num_filters_2', [16, 32, 64, 128]),
            'kernel_size_1': trial.suggest_categorical('kernel_size_1', [(3, 1), (5, 1), (7, 1)]),
            'kernel_size_2': trial.suggest_categorical('kernel_size_2', [(3, 1), (5, 1), (7, 1)]),
            'lstm_units': trial.suggest_categorical('lstm_units', [16, 32, 64, 128]),
            'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
            'dense_units': trial.suggest_categorical('dense_units', [32, 64, 128]),
            'l2_regularization': trial.suggest_loguniform('l2_regularization', 1e-6, 1e-2),
            'pooling_size': trial.suggest_categorical('pooling_size', [(2, 1), (2, 2), (3, 1)])
        }
        print(model_config)
        
        learning_rate = model_config['lr']
        num_filters_1 = model_config['num_filters_1']
        num_filters_2 = model_config['num_filters_2']
        kernel_size_1 = model_config['kernel_size_1']
        kernel_size_2 = model_config['kernel_size_2']
        lstm_units = model_config['lstm_units']
        dropout_rate = model_config['dropout_rate']
        dense_units = model_config['dense_units']
        l2_regularization = model_config['l2_regularization']
        pooling_size = model_config['pooling_size']

        batch_size = config['windowing']['batch_size']
        window_size_seconds = config['windowing']['window_size_seconds']
        step_size_seconds = config['windowing']['step_size_seconds']
        segment_length_seconds = config['segmentation']['segment_length_seconds']

        # Directory paths for training, validation, and testing CSV files
        train_directory_path = './data/train/'
        val_directory_path = './data/val/'

        train_windowed_data, steps_per_epoch_train,input_shapes = process_each_file(train_directory_path, config)
        val_windowed_data, steps_per_epoch_val,_ = process_each_file(val_directory_path, config)


        train_dataset = process_and_create_datasets(train_windowed_data, config)
        val_dataset = process_and_create_datasets(val_windowed_data, config)

        train_dataset = train_dataset.shuffle(1000, seed=random_seed).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.shuffle(1000, seed=random_seed).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)


        for data in train_dataset.take(1):
            print(data)
            features, label = data
            for i, feature_set in enumerate(features):
                print(f"Features shape for sensor {i+1}:", feature_set.numpy().shape)
            print("Label:", label.numpy())
            break

        num_classes = 2  # Adjust as per your number of classes


        model = build_combined_model(input_shapes, num_classes,model_config)
        model.summary()

        class_weight_dict = {0: 2, 1: 1}

        checkpoint = ModelCheckpoint(
            filepath=(
                f"models/optuna4/best_model_"
                f"lr{learning_rate:.6f}_"
                f"filters1_{num_filters_1}_"
                f"filters2_{num_filters_2}_"
                f"kernel1_{kernel_size_1[0]}x{kernel_size_1[1]}_"
                f"kernel2_{kernel_size_2[0]}x{kernel_size_2[1]}_"
                f"lstm_{lstm_units}_"
                f"dropout{dropout_rate:.2f}_"
                f"dense_{dense_units}_"
                f"l2_{l2_regularization:.6f}_"
                f"pooling_{pooling_size[0]}x{pooling_size[1]}.h5"
            ),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )


        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        #pruning_callback = TFKerasPruningCallback(trial, 'val_loss', n_warmup_steps=10)

        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=20, 
                                       restore_best_weights=True,  
                                       verbose=1)
        
        garbage_collection_callback = GarbageCollectionCallback()


    #     # Train the model
        history = model.fit(
            train_dataset,
            epochs=300,  
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            callbacks=[checkpoint, tensorboard_callback, early_stopping,garbage_collection_callback],
            class_weight=class_weight_dict,
            verbose=1

        )
        val_loss = history.history['val_loss'][-1]
        return val_loss
    
    except Exception as e:
        print(f"Trial failed due to: {e}")
        # You can either return a high loss or use `np.inf` to indicate failure.
        return np.inf

@tf.function
def train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train, class_weights):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, logits)
        
        # Apply class weights
        weighted_loss = loss_value * tf.gather(class_weights, tf.cast(y_batch_train, tf.int32))

    # Backpropagation and optimize
    grads = tape.gradient(weighted_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value, logits   
# Custom Training Loop
def custom_train_loop(model, optimizer, loss_fn, train_dataset, val_dataset, epochs, batch_size, steps_per_epoch_train, steps_per_epoch_val, class_weight_dict):
    # Initialize metrics
    class_weights = tf.convert_to_tensor([class_weight_dict[0], class_weight_dict[1]], dtype=tf.float32)
  
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    train_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()
    
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch+1}")
        
        # Reset metrics at the start of each epoch
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        train_loss_metric.reset_states()
        val_loss_metric.reset_states()

        # Iterate over the training batches
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset.take(steps_per_epoch_train)):
            # Call the compiled train step
            loss_value, logits = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train, class_weights)

            # Update training metrics
            train_acc_metric.update_state(y_batch_train, logits)
            train_loss_metric.update_state(loss_value)
            
            # Display current step (overwriting the line)
            print(f"\rStep {step+1}/{steps_per_epoch_train} - Training Loss: {train_loss_metric.result().numpy():.4f}, Accuracy: {train_acc_metric.result().numpy():.4f}", end="")

        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result().numpy()
        train_loss = train_loss_metric.result().numpy()
        print(f"Training accuracy: {train_acc:.4f}, Training loss: {train_loss:.4f}")
        
        # Run validation loop
        for x_batch_val, y_batch_val in val_dataset.take(steps_per_epoch_val):
            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            
            # Update validation metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
            val_loss_metric.update_state(val_loss_value)

        val_acc = val_acc_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()
        print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")

        # Garbage collection and clearing session to release memory
        gc.collect()
        tf.keras.backend.clear_session()
        
def objective(trial, config):
    random_seed = 100
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    
    # TUNE WITH OPTUNA
    config['windowing']['window_size_seconds'] = trial.suggest_int('window_size_seconds', 30, 240, step=30)
    config['windowing']['step_size_seconds'] = 5
    
    config['windowing']['batch_size'] = trial.suggest_int('batch_size', 32, 64, step=32)
    
    learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    zero_weight = 2#trial.suggest_int('zero_weight', 1, 10, step=1)
    
    
    batch_size = config['windowing']['batch_size']
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    segment_length_seconds = config['segmentation']['segment_length_seconds']
    
    
    # Directory paths for training, validation, and testing CSV files
    train_directory_path = './data/test/'
    val_directory_path = './data/test/'
    
    # Cache paths
    train_cache_path = './cache/train_cache'
    val_cache_path = './cache/val_cache'
    
    train_windowed_data, steps_per_epoch_train,input_shapes = process_each_file(train_directory_path, config)
    val_windowed_data, steps_per_epoch_val,_ = process_each_file(val_directory_path, config)

    train_dataset = process_and_create_datasets(train_windowed_data, config)
    val_dataset = process_and_create_datasets(val_windowed_data, config)

    train_dataset = train_dataset.shuffle(1000, seed=random_seed).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.shuffle(1000, seed=random_seed).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    
    #train_dataset = train_dataset.batch(batch_size).repeat()#.prefetch(tf.data.experimental.AUTOTUNE)
    #val_dataset = val_dataset.batch(batch_size).repeat()#.prefetch(tf.data.experimental.AUTOTUNE)


    # FOR THE SigmoidFocalCrossEntropy LOSS
    #train_dataset = train_dataset.map(preprocess_cast)
    #val_dataset = val_dataset.map(preprocess_cast)
    
    for data in train_dataset.take(1):
        print(data)
        features, label = data
        for i, feature_set in enumerate(features):
            print(f"Features shape for sensor {i+1}:", feature_set.numpy().shape)
        print("Label:", label.numpy())
        break

        
    model_config = {
            'lr': learning_rate,
            'num_filters_1': 8,
            'num_filters_2': 16,
            'num_filters_3':32,
            'num_filters_4':64,
            'num_filters_1_sm': 8,
            'num_filters_2_sm': 16,
            'kernel_size_1': (3, 1),
            'kernel_size_2': (3, 1),
            'kernel_size_1_sm': (2, 1),
            'lstm_units': 32,
            'dropout_rate': 0.2,
            'dense_units': 64,
            'l2_regularization': 0.001,
            'pooling_size': (2, 1),
            'num_heads': 8,    # Number of heads in Transformer block
            'ff_dim': 128,     # Feed-forward layer size in Transformer block
        }
    
    #model_file_path = f'models/optuna_fixed/best_model_smallNew_highpass01_order5_PeakRemoval99_ppghighpass02_NewPPGfilter_win{window_size_seconds}_batch{batch_size}_FilterSegment{segment_length_seconds}_zero_weight{zero_weight}_lr{learning_rate}.h5'
    model_file_path = f'models/optuna_fixed/_temp.h5'

    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model = load_model(model_file_path)
    else:
        model = build_combined_model(input_shapes, 2,model_config)
    model.summary()
    
    class_weight_dict = {0: zero_weight, 1: 1}
    
    checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #pruning_callback = TFKerasPruningCallback(trial, 'val_loss', n_warmup_steps=10)

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=20, 
                                   restore_best_weights=True,  
                                   verbose=1)
    
    

#     history = model.fit(
#         train_dataset,
#         epochs=1000,  
#         validation_data=val_dataset,
#         steps_per_epoch=steps_per_epoch_train//batch_size,
#         validation_steps=steps_per_epoch_val//batch_size,
#         #callbacks=[checkpoint, early_stopping],
#         #class_weight=class_weight_dict,
#         verbose=1      
#     )
    epochs = 10 
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    custom_train_loop(model, optimizer, loss_fn, train_dataset, val_dataset, epochs, batch_size, steps_per_epoch_train, steps_per_epoch_val, class_weight_dict)

    val_loss = history.history['val_loss'][-1]
    return val_loss

   
if __name__ == '__main__':
    run_study=False
    config = load_config()
    if run_study:
          # Load configuration
        #study_file = 'optuna_studies/Window_batch_segment_lr_optimization_study.pkl'
        #study = joblib.load(study_file)
        

        #study_name = "Window_batch_segment_lr_optimization_study"  # Choose a name for your study

        #study_name = "Model_optimization_study"
        #study_name = "Window_batch_segment_lr_optimization_study_newModelConfig2"
        study_name = "Window_batch_lr_smallModel_WithTemp_optimization_study"


        # Create the Optuna study
        study = optuna.create_study(direction='minimize')

        # Optimize the objective function with the config passed in
        study.optimize(lambda trial: objective(trial, config), n_trials=20)

        print('Best trial:')
        trial = study.best_trial

        print(f'  Value: {trial.value}')
        print(f'  Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')


        joblib.dump(study, "optuna_studies/"+study_name+".pkl")

    else:
        # Fixed parameters
        fixed_params = {
            'window_size_seconds': 60,
            'segment_length_seconds': 30,
            'batch_size':32,
            'learning_rate':0.0001,
            'zero_weight':2
        }

        # Call the objective function manually
        fixed_result = objective(FixedTrial(fixed_params),config)

        print(f'Objective function result with fixed parameters: {fixed_result}')