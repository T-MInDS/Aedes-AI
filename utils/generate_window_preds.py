import tensorflow as tf
import tensorflow.keras.backend as K

import os, json
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from itertools import chain
from glob import glob
from scipy.signal import savgol_filter
import sys
sys.path.append("./utils")
sys.path.append("./models")
from generate_predictions import *
import models

def main():
    config="./models/Configs/gru_config.json"
    model_files=[f"./models/saved_models/lstm_model_{n}.h5" for n in (15, 30, 45, 60, 75, 90, 105, 120)]#, 135, 150, 165, 180)]
    
    data_files=['./data/train_data.pd','./data/val_data.pd','./data/test_data.pd']

    with open(config) as fp:
        config = json.load(fp)
    data_shape=config["data"]["data_shape"]

    for data_file in data_files:
        data=pd.read_pickle(data_file)
        if "train" in data_file:
            istrain=1
            marker_name="Train"
        elif "test" in data_file:
            istrain=0
            marker_name="Test"
        else:
            istrain=0
            marker_name="Val"
        for i, model_file in enumerate(model_files):
            data_shape[0] = 15 * (i + 1)
            to_save=list()
            if os.path.exists(model_file):
                model = tf.keras.models.load_model(model_file, custom_objects={"r2_keras":r2_keras})
                if istrain==1:
                    results,scaler=gen_preds(model, data, data_shape, None, fit_scaler=True)
                else:
                    results = gen_preds(model, data, data_shape, scaler, fit_scaler=False)

                marker = list()
                model_name=model_file.split('/')[-1].split('.')[0]
                for i in range(len(results)):
                    marker.append(model_name)
                marker=np.asarray(marker)
                to_save.append(np.concatenate([np.reshape(marker, (len(marker),1)), results],axis=1))
            to_save=np.asarray(list(chain(*to_save)))
            to_save=pd.DataFrame(to_save, columns=['Model','County','Year','Month','Day','MoLS','Neural Network'])
            to_save.to_csv("./results/Raw/"+marker_name+"_"+model_name+"_predictions.csv",index=False)


if __name__ == '__main__':
    main()