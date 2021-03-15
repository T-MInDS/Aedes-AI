import tensorflow as tf
import tensorflow.keras.backend as K
import argparse, os, json
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from models import models, visuals

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(14)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='The json configuration file for the aedes model.')
    parser.add_argument('-t', '--testing', action='store_true', help='Whether or not to run the test set.')
    parser.add_argument('-l', '--load', action='store_true', help='Whether or not to load a previously defined model.')
    return parser.parse_args()

def format_data(data, data_shape, samples_per_city, scaler = None, fit_scaler = False, double_peak_multiplier=1):
    data.columns = range(0, len(data.columns))
    if fit_scaler:
        scaler.fit(data.iloc[:, -(data_shape[1] + 1):])
    groups = data.groupby(by = 0)
    data = []
    double_peak_cities = set(pd.read_csv(os.path.expanduser('./data/double_peak.csv'), squeeze=True, index_col=0))
    for city, subset in groups:
        multiplier = 1
        if city in double_peak_cities:
            multiplier = double_peak_multiplier
        random_indices = np.random.randint(0, len(subset) - (data_shape[0] + 1), size = multiplier * samples_per_city)
        for i in range(len(random_indices)):
            random_index= random_indices[i]
            data.append(scaler.transform(subset.iloc[random_index: random_index + data_shape[0], -(data_shape[1] + 1):].values))
    return np.array(data) if not fit_scaler else (np.array(data), scaler)

def split_and_shuffle(data):
    permutation = np.random.permutation(len(data))
    return data[permutation, :, :-1], data[permutation, -1, -1]

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def main():
    args = parse_args()
    with open(args.config) as fp:
        config = json.load(fp)

    for key, value in config['files'].items():
        config['files'][key] = value.replace('/', '\\')

    # load the model as necessary
    model_file = os.path.expanduser(config['files']['model'])
    if args.load and os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file, custom_objects = {'r2_keras': r2_keras})
        print('MODEL LOADED')
    else:
        model = getattr(models, config['model'])(config['data']['data_shape'])
     
    # get the data
    training = pd.read_pickle(os.path.expanduser(config['files']['training']))
    validation = pd.read_pickle(os.path.expanduser(config['files']['validation']))
    testing = pd.read_pickle(os.path.expanduser(config['files']['testing']))
    training, scaler = format_data(training, config['data']['data_shape'], config['data']['samples_per_city'],
                                   scaler=MinMaxScaler(), fit_scaler=True,
                                   double_peak_multiplier=config['data']['double_peak_multiplier'])
    validation = format_data(validation, config['data']['data_shape'], config['data']['samples_per_city'],
                             scaler=scaler)
    testing = format_data(testing, config['data']['data_shape'], config['data']['samples_per_city'],
                          scaler=scaler)
        
    if config['data']['temperature_augmentation']==True:
        temp_training = np.random.randint(0, len(training) - 1, size = 30000)
        hi_temp = np.copy(training[temp_training[:15000]])
        lo_temp = np.copy(training[temp_training[15000:]])
        for i in range(0,len(hi_temp)):
            #scaler.transform for 41C is 0.87
            #scaler.transform for 3C is 0.33
            #might want to change this from uniform?
            hi_avg=np.random.uniform(0.87,1)
            lo_avg=np.random.uniform(0,0.33)

            lo_shift=lo_avg - np.average(np.concatenate((lo_temp[i,:,0],lo_temp[i,:,1])))
            lo_temp[i,69:,:2]+=lo_shift
            lo_temp[i,69:,-1]=0
            
            hi_shift=hi_avg - np.average(np.concatenate((hi_temp[i,:,0],hi_temp[i,:,1])))
            hi_temp[i,69:,:2]+=hi_shift
            hi_temp[i,69:,-1]=0
        training = np.concatenate([training, hi_temp, lo_temp])

    X_train, y_train = split_and_shuffle(training)
    X_val, y_val = split_and_shuffle(validation)
    X_test, y_test = split_and_shuffle(testing)

    if args.testing:
        print('Running test set...')
        model.evaluate(X_test, y_test)

    else:
        model.compile(optimizer = getattr(tf.keras.optimizers, config['compile']['optimizer'])(lr = config['compile']['learning_rate']),
                      loss = config['compile']['loss'], metrics = [r2_keras])
        history = model.fit(X_train, y_train, validation_data = (X_val, y_val), **config['fit'],
                  callbacks = [tf.keras.callbacks.TensorBoard(), tf.keras.callbacks.EarlyStopping(patience = 30, restore_best_weights = True)])
        model.save(model_file, save_format = 'h5')

        visuals.plot_loss(history)
        visuals.plot_r2(history)


if __name__ == '__main__':
    main()