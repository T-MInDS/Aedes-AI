import tensorflow as tf
import tensorflow.keras.backend as K
import argparse, os, json
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append("./models")
sys.path.append("./")
import models, visuals
from utils.match_peaks import peak_finder

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2*1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

np.random.seed(14)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='The json configuration file for the aedes model.')
    parser.add_argument('-t', '--testing', action='store_true', help='Whether or not to run the test set.')
    parser.add_argument('-l', '--load', action='store_true', help='Whether or not to load a previously defined model.')
    return parser.parse_args()

def format_data(data, data_shape, samples_per_city, scaler=None, fit_scaler=False,
                summer_samples=0, winter_samples=0, summer_cities=set(), winter_cities=set()):
    data.columns = range(0, len(data.columns))
    if fit_scaler:
        scaler.fit(data.iloc[:, -(data_shape[1] + 1):])
    groups = data.groupby(by = 0)
    data = []
    for city, subset in groups:
        random_indices = np.random.randint(0, len(subset) - (data_shape[0] + 1), size = samples_per_city)

        if (summer_samples and city in summer_cities) or (winter_samples and city in winter_cities):
            peaks = peak_finder(subset.values[:, -1] / subset.values[:, -1].max(), 0.2, 7, 7)
            season_intervals = []
            start = 0
            for i, peak in enumerate(peaks):
                if peak[0] > (365 * (1 + len(season_intervals))):
                    # peak[0] is next season
                    season_intervals.append((max(peaks[start][0], data_shape[0]), peaks[i-1][1]))
                    start = i
            # add final season
            season_intervals.append((peaks[start][0], peaks[-1][1]))
            summer_indices = np.concatenate([range(*szn) for szn in season_intervals]).astype(int) - data_shape[0]
            if summer_samples and city in summer_cities:
                random_indices = np.concatenate([random_indices, np.random.choice(summer_indices, size = summer_samples)])
            if winter_samples and city in winter_cities:
                all_indices = set(np.arange(len(subset) - data_shape[0], dtype=int))
                winter_indices = np.array(list(all_indices.difference(summer_indices)), dtype=int)
                random_indices = np.concatenate([random_indices, np.random.choice(winter_indices, size = winter_samples)])

        for i in range(len(random_indices)):
            random_index = random_indices[i]
            sample = scaler.transform(subset.iloc[random_index: random_index + data_shape[0], -(data_shape[1] + 1):].values)
            data.append(sample)
    return np.array(data, dtype=np.float32) if not fit_scaler else (np.array(data, dtype=np.float32), scaler)

def split_and_shuffle(data):
    print(data.shape)
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
     
    summer_cities = set(pd.read_csv(os.path.expanduser('./data/double_peak.csv'), squeeze=True, index_col=0))
    winter_cities = {"Dane,Wisconsin", "Milwaukee,Wisconsin", "New Haven,Connecticut", "Bronx,New York", "Kings,New York",
                     "Monmouth,New Jersey", "Mono,California", "Monterey,California", "Morris,New Jersey", "Napa,California",
                     "Nassau,New York", "New Hanover,North Carolina", "New River,Arizona", "Okaloosa,Florida", "Orange,California",
                     "Oro Valley,Arizona", "Prescott,Arizona", "Rio Rico,Arizona", "Rockland,New York", "Sacramento,California"}

    # get the data
    training = pd.read_pickle(os.path.expanduser(config['files']['training']))
    validation = pd.read_pickle(os.path.expanduser(config['files']['validation']))
    testing = pd.read_pickle(os.path.expanduser(config['files']['testing']))
    training, scaler = format_data(training, config['data']['data_shape'], config['data']['samples_per_city'],
                                   scaler=MinMaxScaler(), fit_scaler=True,
                                   summer_samples=config['data']['summer_samples'],
                                   winter_samples=config['data']['winter_samples'],
                                   summer_cities=summer_cities,
                                   winter_cities=winter_cities)
    validation = format_data(validation, config['data']['data_shape'], config['data']['samples_per_city'],
                             scaler=scaler)
    testing = format_data(testing, config['data']['data_shape'], config['data']['samples_per_city'],
                          scaler=scaler)
   
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
                  callbacks = [tf.keras.callbacks.TensorBoard(), tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True)])
        model.save(model_file, save_format = 'h5')

        visuals.plot_loss(history, args.config.split('.')[0].split('/')[-1])
        visuals.plot_r2(history, args.config.split('.')[0].split('/')[-1])


if __name__ == '__main__':
    main()