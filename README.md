# Aedes-AI

This repository implements the Aedes-AI project, a machine learning application based on the Mosquito Landscape Simulation (MoLS).

The code in this repository is organized in six folders. All code is set up to run from the home directory (the same directory as the ReadMe file).

## data
This folder contains pickled pandas dataframes for the training, validation, and testing sets.
Additionally, there is a csv listing the cities that are considered "Double Peak" cities.

## figures
This folder contains code for no-geographic visualizations.

## map
This folder contains code for geographic visualizations.

## models
This folder contains all code required to train, test, and save models.
Each model has an associated config file in .json format in models/configs.
The configuration file contains associated information about model parameters, the type of model, and data paths.
To train a model, run the training.py file with command line arguments:

  **python models/training.py <config filepath>**
  
To test the same model, run the training code with extra arguments:

  **python models/training.py <config filepath> --load --test**
  
This will load and test the previously trained model. Note that the filepath to the model is in the config file.
To train all previously constructed models, the training.sh shell script can be used.

Configuration files are in the following format (using models/configs/lstm_config.json as an example):  
  {  
      "model": "lstm_model", # model name from the models/models.py file  
      "data": {  
          "data_shape": [90, 4], # data shape of the time series  
          "samples_per_city": 1000, # the number of samples to randomly take from a location for training  
          "double_peak_multiplier": 1, # the oversampling rate for double peak cities (DPO models)  
          "temperature_augmentation": false # whether or not temperature augmentation should be included (TA models)  
      },  
      "compile": {  
          "optimizer": "Adam", # optimizer for training  
          "learning_rate": 0.0001, # initial learning rate  
          "loss": "mse" # loss function  
      },  
      "fit": {  
          "batch_size": 64, # batch size  
          "epochs": 100 # number of training epochs  
      },  
      "files": {  
          "training": "./data/train_data.pd", # filepath to the training data  
          "validation": "./data/val_data.pd", # filepath to the validation data  
          "testing": "./data/test_data.pd", # filepath to the testing data  
          "model": "./models/saved_models/lstm_model.h5" # filepath to save/load the model  
      }  
  }  

## results
This folder contains previously run model predictions and outputs, both smoothed and unsmoothed (raw) for all datasets.

## utils
This folder contains all utility code and code used to construct and define metrics.
Code used to generate results and run models to obtain output is found here.
