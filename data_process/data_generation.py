# Importing required libraries
import logging
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

# Static variables
RANDOM_STATE = 42

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = os.getenv('CONF_PATH') if you have problems with env variables
CONF_FILE = "settings.json"
# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for generating Iris dataset
@singleton
class IrisDatasetGenerator():
    def __init__(self):
        self.df_train = None
        self.df_inference = None
        
    def load_data(self):
        # Load iris dataset from scikit-learn library
        iris = load_iris()
        data = np.c_[iris.data, iris.target]
        columns = np.append(iris.feature_names, ["target"])
        iris_full_df = pd.DataFrame(data, columns=columns)
        logger.info(f"Loading iris data ...")   
        return iris_full_df
    
    def split_data(self, iris_full_df, test_size=0.2, random_state=RANDOM_STATE):
        # Split the data into train and test sets
        train_data, inference_data = train_test_split(iris_full_df, test_size=test_size, random_state=random_state)
        self.df_train = train_data
        self.df_inference = inference_data.drop(columns=['target'])
        logger.info(f"Splitting iris data ...")   

    def save_datasets(self, train_filepath, inference_filepath):
        # Save train and inference datasets to CSV files
        self.df_train.to_csv(train_filepath, index=False)
        logger.info(f"Saving train data to {train_filepath}...")
        self.df_inference.to_csv(INFERENCE_PATH, index=False)
        logger.info(f"Saving inference data to {inference_filepath}...")    

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    iris_dataset_generator = IrisDatasetGenerator()
    iris_full_df = iris_dataset_generator.load_data()
    iris_dataset_generator.split_data(iris_full_df, test_size=0.2, random_state=42)
    iris_dataset_generator.save_datasets(TRAIN_PATH, INFERENCE_PATH)
    logger.info("Script completed successfully.")