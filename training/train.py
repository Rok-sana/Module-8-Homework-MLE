"""
This script prepares the data, runs the training, and saves the model.
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import time
from datetime import datetime
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np

# # Using specialized MLFlow for Tensorflow 
import mlflow
mlflow.autolog()

# Adds the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
#CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = "settings.json"

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")



class DataProcessor():
    def __init__(self):
        pass

    def prepare_data(self, max_rows=None):
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path):
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    def data_rand_sampling(self, df, max_rows):
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df



class  NeuralNetworkTraining():
    #Handles model training and evaluation

    def __init__(self) -> None:
        """Initializes the neural network model """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense(3, activation='softmax')  # Output will be for 3 classes
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_evaluate_model(self, training_data: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        """Runs the model training and evaluation process"""
        logging.info("Running training...")
    
        X_train, X_test, y_train, y_test = self.data_split(training_data, test_size=test_size)

        # Build the model explicitly before training
        self.model.build(input_shape=(X_train.shape))

        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")

        self.test(X_test, y_test) 
        self.save(out_path)

    def data_split(self, data: pd.DataFrame, test_size: float = 0.33) -> tuple:
        """Splitting data on features and target"""
        features = data[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
        target = data.target.values
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=conf['general']['random_state'])
        logging.info("Splitting data into features and target...")
        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Prepare target in categorical and training the model"""
        y_train_cat = tf.keras.utils.to_categorical(y_train)
        logging.info("Fitting model...")
        self.model.fit(X_train, y_train_cat, epochs=100, batch_size=32)  # Used examples of number of epochs and batch

    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """Evaluates the model on the test data"""
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        logging.info(f"Accuracy score: {accuracy}")
        return accuracy

    def save(self, path):
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, 'sample_model.keras')
        else:
            path = os.path.join(MODEL_DIR, path)

        self.model.save(path, save_format='tf')


def main():
    configure_logging()

    data_proc = DataProcessor()
    traiinig = NeuralNetworkTraining()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    traiinig.train_evaluate_model(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()