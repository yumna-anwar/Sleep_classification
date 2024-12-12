import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json
from Data_loader_new import *
from model import *
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter 
from sklearn.metrics import f1_score,precision_recall_curve


if __name__ == '__main__':
    
    mod_name = 'best_model_smallNew2_highpass01_order5_NoPeakRemoval_ppghighpass02low5_RobScaleAll_BinFocalLossG4a02_lrdecayCos12k_win60_batch32_FilterSegment30_lr0.0001'
    saved_model_dir = "wrapped_model"
    
#     model = tf.keras.models.load_model('models/folds/'+mod_name+'.h5')
    
#     # Save the model in the SavedModel format
#     model.save(saved_model_dir, save_format="tf")
#     print(f"Model saved to {saved_model_dir} in SavedModel format.")

    # Convert SavedModel to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False

    # Convert and save the TFLite model
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Model successfully converted to TFLite format.")
    

    
    
    


