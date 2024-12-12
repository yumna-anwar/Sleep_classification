import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from datetime import datetime
from Data_loader_new import *
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

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/ebuild/software/CUDA/11.7.0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.random.set_seed(1)
class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()
        

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

    window_size_ppg = int(window_size_seconds * config['frequencies']['ppg'])
    step_size_ppg = int(step_size_seconds * config['frequencies']['ppg'])

    input_shapes = (window_size_ppg, 9, 1)
    
    
    train_directory_path = ['./data/5folds/fold3/','./data/5folds/fold2/','./data/5folds/fold1/']
    val_directory_path = ['./data/5folds/fold4/']


    train_dataset = tf.data.Dataset.from_generator(
        lambda: process_each_file_StackedModel(train_directory_path, config),
        output_types=( tf.float32, tf.int64),
        output_shapes=((None, None, 1), ())
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: process_each_file_StackedModel(val_directory_path, config),
        output_types=( tf.float32, tf.int64),
        output_shapes=((None, None, 1), ())
    )
      
    train_dataset = train_dataset.cache().shuffle(800).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache().shuffle(800).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
        
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
            'loss_gamma': 4,
            'loss_alpha': 0.2,
            'lr_alpha': 0.1,
             'lr_decay_steps': 12100
        }
    
    #with strategy.scope():
    model_file_path = f'models/folds_stacked/best_model_NoTemp_removedNS_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12ka01_win{window_size_seconds}_step{step_size_seconds}_batch{batch_size}_FilterSegment{segment_length_seconds}_lr{learning_rate}.h5'
    
    #model_file_path = f'models/folds/test.h5'

    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model = load_model(model_file_path)
    else:
        model = build_stacked_model(input_shapes, 2,model_config)

    model.summary()

    #class_weight_dict = {0: zero_weight, 1: 1}
    #class_weight_dict = {0: 3.75, 1: 0.57}

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
        epochs=500,  
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, garbage_collection_callback],
        verbose=1      
    )
    
    
    val_loss = history.history['val_loss'][-1]
        
    tf.keras.backend.clear_session()
    
 
    return val_loss

  
if __name__ == '__main__':
    run_study=False
    
    config = load_config()
    if run_study:
         
        study_name = "Stacked_Model"

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
        #tf.profiler.experimental.stop()