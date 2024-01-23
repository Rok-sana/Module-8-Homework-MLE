# Module-8-Homework-MLE

Welcome to the Module-8-Homework-MLE project. This project serves as an organized template for Machine Learning development and deployment. The main goal is to familiarize with ML model deployment tools, as part of the homework task . The project covers all aspects of the ML pipeline, including data processing, model training and inference on new data.


# Cloning from GitHub
TClick the 'Code' button on your forked repository, copy the provided link, and employ the `git clone` command in your terminal followed by the copied link.


# Project Structure

This project adopts a modular structure, where each folder serves a specific purpose.

```
Module-8-Homework-MLE
├── data                      # Data files used for training and inference
│   ├── iris_inference_data.csv
│   └── iris_train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes used in scripts
├── run_ml_pipeline.sh        # Script for running the ML pipeline
├── settings.json             # Configurable parameters and settings
└── README.md

```
# Settings

Please create a .env file or manually initializing an environment variable as `CONF_PATH=settings.json`.


# Data, Training,  Inference:
For convenient use of the project was added `run_ml_pipeline.sh` script that is responsible for data generation, model training and testing, as well as checking on the inference data.

To execute this script, you would typically open a terminal, navigate to the directory containing the script, and then run the following command:

`bash run_ml_pipeline.sh`

or
`./run_ml_pipeline.sh` 

Make sure the script has the necessary execution permissions. If not, you can grant them using:
`chmod +x run_ml_pipeline.sh`

Here's a general overview of what the script does:

- Error Handling:
The script starts with set -e, which means that it will exit immediately if any command it runs returns a non-zero exit status, indicating an error.

- Function Definitions:
create_directory(): A function that takes a directory path as an argument, checks if the directory exists, and creates it if it doesn't.
build_and_run_docker(): A function that builds and runs a Docker container based on the provided Dockerfile, image name, and build arguments. It also waits for the container to finish running and then copies specific files from the container to the local machine.

- Create Directories:
Calls create_directory twice to ensure that the "data" and "models" directories exist. If they don't, the script creates them.

- Build and Run Training Docker Container:
Calls build_and_run_docker to build and run a Docker container for training the ML model. It specifies the Dockerfile, image name, and settings for the build.

- Build and Run Inference Docker Container and create result data:
Calls build_and_run_docker again, this time for the inference phase. It uses a different Dockerfile, sets a unique image name, and specifies both the settings and a sample model name ("sample_model.keras") as build arguments.

