import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from Data_loader import *
from model import *
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback
import joblib
from tensorflow.keras.callbacks import EarlyStopping


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

def objective(trial, config):
    random_seed = 100
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    label_col = config['labels']['label_column']
    
    model_config = {
            'lr': 0.00001,
            'num_filters_1': 32,
            'num_filters_2': 64,
            'kernel_size_1': (3, 1),
            'kernel_size_2': (3, 1),
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'dense_units': 128,
            'l2_regularization': 1e-6,
            'pooling_size': (3, 1)
        }
    
    # TUNE WITH OPTUNA
    config['windowing']['window_size_seconds'] = 240#trial.suggest_int('window_size_seconds', 60, 300, step=30)
    config['windowing']['step_size_seconds'] = config['windowing']['window_size_seconds']//2
    
    config['segmentation']['segment_length_seconds'] = 60#trial.suggest_int('segment_length_seconds', 20, 100, step=20)
    config['segmentation']['step_size_seconds'] = config['segmentation']['segment_length_seconds']//4
    config['synthetic_data']['duration'] = config['segmentation']['segment_length_seconds']
    
    config['windowing']['batch_size'] = 32#trial.suggest_int('batch_size', 32, 64, step=32)
    
    learning_rate = 0.0001#trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    zero_weight = 4#trial.suggest_int('zero_weight', 1, 10, step=1)
    
    
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

    model = build_combined_model(input_shapes, 2,model_config)
    model.summary()
    
    class_weight_dict = {0: zero_weight, 1: 1}
    
    checkpoint = ModelCheckpoint(f'models/optuna6/best_model_win{window_size_seconds}_batch{batch_size}_FilterSegment{segment_length_seconds}_zero_weight{zero_weight}_lr{learning_rate}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #pruning_callback = TFKerasPruningCallback(trial, 'val_loss', n_warmup_steps=10)

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=20, 
                                   restore_best_weights=True,  
                                   verbose=1)
    
    

    history = model.fit(
        train_dataset,
        epochs=1000,  
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch_train,
        validation_steps=steps_per_epoch_val,
        callbacks=[checkpoint, tensorboard_callback, early_stopping],
        class_weight=class_weight_dict,
        verbose=1      
    )
    val_loss = history.history['val_loss'][-1]
    return val_loss

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


    #     # Train the model
        history = model.fit(
            train_dataset,
            epochs=300,  
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            callbacks=[checkpoint, tensorboard_callback, early_stopping],
            class_weight=class_weight_dict,
            verbose=1

        )
        val_loss = history.history['val_loss'][-1]
        return val_loss
    
    except Exception as e:
        print(f"Trial failed due to: {e}")
        # You can either return a high loss or use `np.inf` to indicate failure.
        return np.inf
    
if __name__ == '__main__':
      # Load configuration
    #study_file = 'optuna_studies/Window_batch_segment_lr_optimization_study.pkl'
    #study = joblib.load(study_file)
    config = load_config()
    
    #study_name = "Window_batch_segment_lr_optimization_study"  # Choose a name for your study
    
    #study_name = "Model_optimization_study"
    #study_name = "Window_batch_segment_lr_optimization_study_newModelConfig2"
    study_name = "train1"
    

    # Create the Optuna study
    study = optuna.create_study(direction='minimize')

    # Optimize the objective function with the config passed in
    study.optimize(lambda trial: objective(trial, config), n_trials=1)

    print('Best trial:')
    trial = study.best_trial

    print(f'  Value: {trial.value}')
    print(f'  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
        
        
    joblib.dump(study, "optuna_studies/"+study_name+".pkl")