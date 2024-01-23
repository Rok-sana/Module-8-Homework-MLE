# Module-8-Homework-MLE

Welcome to the Module-8-Homework-MLE project. This project serves as an organized template for Machine Learning development and deployment. The main goal is to familiarize with ML model deployment tools, as part of the homework task . The project covers all aspects of the ML pipeline, including data processing, model training and inference on new data.

# Prerequisites

Before delving into the detailed setup and utilization of this project, ensure that your local development environment meets specific prerequisites. These prerequisites guarantee that your local setup can efficiently run and support the project. If you encounter challenges installing Docker Desktop or MLFlow, you have the option to skip these steps, modify your code, and work directly on your local machine using Python and Git.

# Forking and Cloning from GitHub
To initiate the use of this project, you must first create a copy on your GitHub account by 'forking' it. On the main page of the Module-8-Homework-MLE project, click on the 'Fork' button at the top right corner. This action will replicate the project under your account. Subsequently, 'clone' it to your local machine for personal use. Click the 'Code' button on your forked repository, copy the provided link, and employ the git clone command in your terminal followed by the copied link.

# Setting Up Development Environment
It is recommended to set up an Integrated Development Environment (IDE), with Visual Studio Code (VSCode) being a commendable choice. Download VSCode from the official website (https://code.visualstudio.com/Download). Post-installation, open VSCode, navigate to the File menu, and click Add Folder to Workspace. Locate the directory where you cloned the forked repository and add it. VSCode offers extensive support for various programming languages, providing features like syntax highlighting, code completion, and debugging configurations. With this setup, you can seamlessly edit files, traverse your project, and contribute to Module-8-Homework-MLE. For script execution, open a new terminal in VSCode by selecting Terminal -> New Terminal.

# Installing Docker Desktop
Docker Desktop installation is a straightforward process. Visit the Docker official website's download page (Docker Download Page) and select the version corresponding to your operating system. After downloading the installer, execute it and follow the on-screen instructions.

Upon successful installation, confirm Docker Desktop's proper functioning by opening the application. It typically appears in your applications or programs list. Docker Desktop remains idle until you execute Docker commands. The application facilitates the management of containers, images, and networks directly from your desktop, simplifying operations. Note that Docker requires virtualization to be enabled in your system's BIOS settings. If issues arise, check your virtualization settings or consult Docker's installation troubleshooting guide. Now, you're prepared to work with Dockerized applications.

Installing MLFlow 
MLFlow installation on a Windows local machine is achievable using pip, the Python package installer. Open the command prompt (search for cmd in the Start menu) and input the following command:

`pip install mlflow`

Upon successful installation, you can commence managing and deploying your ML models with MLFlow. For detailed information on optimizing MLFlow usage, refer to the official MLFlow documentation or use the mlflow --help command.

If installation issues arise, consider bypassing them by commenting out the corresponding lines in the train.py and requirements.txt files.

To run MLFlow, type mlflow ui in your terminal and press enter. If that doesn't work, try python -m mlflow ui. This initiates the MLFlow tracking UI, typically accessible on your localhost at port 5000. Access the tracking UI through your web browser at http://localhost:5000.

Project Structure

This project adopts a modular structure, where each folder serves a specific purpose.

```
Module-8-Homework-MLE
├── data                      # Data files used for training and inference (can be generated with data_generation.py script)
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

The project's configurations are managed using the settings.json file, storing essential variables that control project behavior. Examples include paths to resource files, constant values, hyperparameters, or environment-specific settings. Before running the project, ensure that all paths and parameters in settings.json are correctly defined. You may need to pass the config path to your scripts, either by creating a .env file or manually initializing an environment variable as CONF_PATH=settings.json.

Note: Some IDEs, including VSCode, may face issues detecting environment variables defined in the .env file. If problems persist, attempt to run scripts in debug mode or, as a workaround, hardcode necessary parameters directly into your scripts. Be cautious not to expose sensitive data when sharing or making your code public. For such scenarios, consider using secret management tools provided by your environment.

## Data, Training,  Inference:
For convenient use of the project was added `run_ml_pipeline.sh` script that is responsible for data generation, model training and testing, as well as checking on the inference data.
Here's a general overview of what the script does:

Error Handling:
The script starts with set -e, which means that it will exit immediately if any command it runs returns a non-zero exit status, indicating an error.

Function Definitions:
create_directory(): A function that takes a directory path as an argument, checks if the directory exists, and creates it if it doesn't.
build_and_run_docker(): A function that builds and runs a Docker container based on the provided Dockerfile, image name, and build arguments. It also waits for the container to finish running and then copies specific files from the container to the local machine.

Create Directories:
Calls create_directory twice to ensure that the "data" and "models" directories exist. If they don't, the script creates them.

Build and Run Training Docker Container:
Calls build_and_run_docker to build and run a Docker container for training the ML model. It specifies the Dockerfile, image name, and settings for the build.

Build and Run Inference Docker Container:
Calls build_and_run_docker again, this time for the inference phase. It uses a different Dockerfile, sets a unique image name, and specifies both the settings and a sample model name ("sample_model.keras") as build arguments.

To execute this script, you would typically open a terminal, navigate to the directory containing the script, and then run the following command:

`bash run_ml_pipeline.sh`

or
`./run_ml_pipeline.sh` 

Make sure the script has the necessary execution permissions. If not, you can grant them using:
`chmod +x run_ml_pipeline.sh`