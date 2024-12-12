import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from datetime import datetime
from Data_loader_features import *
from model import *
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from optuna.trial import FixedTrial
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
import gc
from tensorflow.keras import backend as K
import itertools
import random
import threading
import time
import GPUtil
import psutil
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/ebuild/software/CUDA/11.7.0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(1)
class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()
        
def extract_features_and_labels(tf_dataset):
    """
    Convert a TensorFlow dataset into NumPy arrays for Scikit-learn.
    """
    features, labels = [], []
    for batch_features, batch_labels in tf_dataset:
        features.append(batch_features.numpy())
        labels.append(batch_labels.numpy())
    features = np.vstack(features)  # Stack all batches into a single array
    labels = np.hstack(labels)  # Flatten labels into a single array
    return features, labels

@tf.autograph.experimental.do_not_convert
def objective(trial, config):
    random_seed = 2
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    
    # TUNE WITH OPTUNA
    config['windowing']['window_size_seconds'] = trial.suggest_int('window_size_seconds', 30, 240, step=30)
    config['windowing']['step_size_seconds'] = 30
    
    config['windowing']['batch_size'] = trial.suggest_int('batch_size', 32, 64, step=32)
    
    learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    zero_weight = trial.suggest_int('zero_weight', 1, 10, step=1)
    
    
    batch_size = config['windowing']['batch_size']
    window_size_seconds = config['windowing']['window_size_seconds']
    step_size_seconds = config['windowing']['step_size_seconds']
    segment_length_seconds = config['segmentation']['segment_length_seconds']
    
    window_size_acc = int(window_size_seconds * config['frequencies']['accelerometer'])
    step_size_acc = int(step_size_seconds * config['frequencies']['accelerometer'])

    window_size_gyro = int(window_size_seconds * config['frequencies']['gyroscope'])
    step_size_gyro = int(step_size_seconds * config['frequencies']['gyroscope'])

    window_size_ppg = int(window_size_seconds * config['frequencies']['ppg'])
    step_size_ppg = int(step_size_seconds * config['frequencies']['ppg'])

    window_size_temp = int(window_size_seconds * config['frequencies']['temperature'])
    step_size_temp = int(step_size_seconds * config['frequencies']['temperature'])
    
    ppg_input_shape = (window_size_ppg, len(config['features']['ppg']), 1)
    gyro_input_shape = (window_size_gyro, len(config['features']['gyroscope']), 1)
    acc_input_shape = (window_size_acc, len(config['features']['accelerometer']), 1)
    temp_input_shape = (window_size_temp, len(config['features']['temperature']), 1)
    
    
    train_directory_path = ['./data/5folds/fold3/','./data/5folds/fold2/','./data/5folds/fold1/']
    val_directory_path = ['./data/5folds/fold4/']

    train_dataset = load_all_files_as_dataset(train_directory_path, config, batch_size=32, shuffle=True, buffer_size=2000)
    val_dataset = load_all_files_as_dataset(val_directory_path, config, batch_size=32, shuffle=False, buffer_size=10)
    
    #input_dim = 63
    input_dim = 321
    
    model_config = {
            'lr': learning_rate,
        }
    
    
    model_file_path = f'models_features/folds/best_model_features_focalLoss_Robscale_win{window_size_seconds}_batch{batch_size}_lr{learning_rate}.h5'
    

    model = feature_model_1D(input_dim, model_config)
    model.summary()

    class_weight_dict = {0: 3.75, 1: 0.57}

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #pruning_callback = TFKerasPruningCallback(trial, 'val_loss', n_warmup_steps=10)

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=10, 
                                   restore_best_weights=True,  
                                   verbose=1)
    garbage_collection_callback = GarbageCollectionCallback()
    
    history = model.fit(
        train_dataset,
        epochs=100,  
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, garbage_collection_callback],
        #class_weight=class_weight_dict,
        verbose=1      
    )
    
    # Stop monitoring and join thread
    val_loss = history.history['val_loss'][-1]
        
    tf.keras.backend.clear_session()
    
 
    return val_loss
def objective_sklearn(trial, config, model_type="SVM"):
    random_seed = 2
    np.random.seed(random_seed)
    
    features_acc = config['features']['accelerometer']
    features_gyro = config['features']['gyroscope']
    features_ppg = config['features']['ppg']
    features_temp = config['features']['temperature']
    
    # TUNE WITH OPTUNA
    window_size_seconds = config['windowing']['window_size_seconds']
    batch_size = config['windowing']['batch_size']
    
    if model_type == "SVM":
        C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Regularization parameter
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])  # Kernel coefficient for 'rbf'
    elif model_type == "RF":
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)  # Number of trees
        max_depth = trial.suggest_int('max_depth', 5, 50, step=5)  # Depth of each tree
    
    train_directory_path = ['./data/5folds/fold3/','./data/5folds/fold2/','./data/5folds/fold1/']
    val_directory_path = ['./data/5folds/fold4/']

    train_dataset = load_all_files_as_dataset(train_directory_path, config, batch_size=32, shuffle=True, buffer_size=2000)
    val_dataset = load_all_files_as_dataset(val_directory_path, config, batch_size=32, shuffle=False, buffer_size=10)
  
    # Extract features and labels from TensorFlow datasets
    train_features, train_labels = extract_features_and_labels(train_dataset)
    val_features, val_labels = extract_features_and_labels(val_dataset)
    
    os.makedirs("data", exist_ok=True)
    joblib.dump((train_features, train_labels), "data/train_dataset.pkl")
    joblib.dump((val_features, val_labels), "data/val_dataset.pkl")

    ## Select model
    if model_type == "SVM":
        model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=random_seed)
        model_file_path = f'models_features/svm_model_balanced_noStanScale_Robscale_win{window_size_seconds}_batch{batch_size}_C{C:.2f}_gamma{gamma}.pkl'
        
    elif model_type == "RF":
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=max_depth, random_state=random_seed, class_weight='balanced')
        model_file_path = f'models_features/rf_model_balanced_win{window_size_seconds}_batch{batch_size}_estimators{n_estimators}_depth{max_depth}.pkl'

        
    # Train SVM
    model.fit(train_features, train_labels)
    
    
    joblib.dump(model, model_file_path)
    
        # Evaluate the model
    if model_type == "SVM":
        decision_scores = model.decision_function(val_features)
    elif model_type == "RF":
        decision_scores = model.predict_proba(val_features)[:, 1]
    
    auc = roc_auc_score(val_labels, decision_scores)
    print(f"AUC ({model_type}): {auc}")
    
    return auc
  
if __name__ == '__main__':
    run_study=True
    
    config = load_config()
    if run_study:
#         study_name = "Window_batch_lr_smallModel_WithTemp_optimization_study"
#         study = optuna.create_study(direction='minimize')
#         study.optimize(lambda trial: objective(trial, config), n_trials=20)
#         print('Best trial:')
#         trial = study.best_trial

#         print(f'  Value: {trial.value}')
#         print(f'  Params: ')
#         for key, value in trial.params.items():
#             print(f'    {key}: {value}')
#         joblib.dump(study, "optuna_studies/"+study_name+".pkl")

        # Optimize the objective function for Random Forest
        study_name = "RF_study_10trials"
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(lambda trial: objective_sklearn(trial, config, model_type="RF"), n_trials=10)
        print("Best trial (RF):")
        trial_rf = study_rf.best_trial
        print(f"  Value: {trial_rf.value}")
        print(f"  Params: ")
        for key, value in trial_rf.params.items():
            print(f"    {key}: {value}")
        joblib.dump(study_rf, "optuna_studies/" + study_name + "_rf.pkl")

    else:
        # Fixed parameters
#         fixed_params = {
#             'window_size_seconds': 60,
#             'segment_length_seconds': 30,
#             'batch_size':32,
#             'learning_rate':0.00001,
#             'zero_weight':2
#         }

#         # Call the objective function manually
#         fixed_result = objective(FixedTrial(fixed_params),config)
        
        
        fixed_params = {
            'window_size_seconds': 60,
            'segment_length_seconds': 30,
            'batch_size':32,
            'learning_rate':0.0001,
            'zero_weight':2,
            'C':0.001,
            'gamma':'scale'
        }
        fixed_result = objective_sklearn(FixedTrial(fixed_params),config)
        
        print(f'Objective function result with fixed parameters: {fixed_result}')
        #tf.profiler.experimental.stop()