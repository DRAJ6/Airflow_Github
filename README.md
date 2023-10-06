# Advertising Click Prediction

This lab provides an overview of a Python script used to build and evaluate a logistic regression model for predicting whether a user will click on an advertisement. The script includes data loading, preprocessing, model building, and evaluation. This lab is designed to run locally on Apache Airflow

## Overview
This Python script is designed to perform the following tasks:

- **Load Data**: The script loads advertising data from a CSV file using the Pandas library.

- **Data Preprocessing**: Data preprocessing involves cleaning and preparing the data for modeling. It includes splitting the data into training and testing sets, scaling numerical features, and encoding categorical features.

- **Model Building**: A logistic regression model is constructed using scikit-learn's LogisticRegression class. Hyperparameter tuning is performed using a grid search with cross-validation to find the best combination of hyperparameters.

- **Model Evaluation**: The trained logistic regression model is evaluated on the testing data, and the model's accuracy is reported.


## Code Components

#### Functions
- load_data(): Loads the advertising data from a CSV file and returns it as a Pandas DataFrame.

- data_preprocessing(data): Preprocesses the data by splitting it into training and testing sets, scaling numerical features, and applying transformations.

- build_model(data, filename): Builds a logistic regression model using grid search and saves the trained model to a file.

- load_model(data, filename): Loads a saved logistic regression model, evaluates it on test data, and prints the model's score.

### Data Loading

Use the below command to fetch the data from the Google Cloud storage which can reflect the updated version of the data. The data will be fetched from the cloud and stored in the data folder.
```commandline
dvc fetch
```
The script loads the advertising data from a CSV file named "advertising.csv" located in the "../data" directory.

### Data Preprocessing
Data preprocessing includes:

Splitting the data into features (X) and the target variable (y).
Scaling numerical features using MinMaxScaler and StandardScaler.
Combining feature transformations using a column transformer.
Splitting the data into training and testing sets.
### Model Building
A logistic regression model is created and hyperparameter tuning is performed using grid search with cross-validation. Hyperparameters considered include penalty, C, class weight, and solver. The built model will stored locally in the model folder.

### Model Evaluation
The trained logistic regression model is loaded, evaluated on the testing data, and the model's accuracy on the test set is printed.


## DAG
This Python script defines an Apache Airflow DAG (Directed Acyclic Graph) named 'Airflow-GitHub' that automates a data processing and machine learning workflow. It utilizes PythonOperators to execute various tasks sequentially.

#### Import Necessary Libraries and Modules:

Import the required modules and libraries from Apache Airflow and other Python modules.
Set up configurations for Airflow, such as enabling XCom pickling for passing data between tasks.
#### Define GitHub Repository Information:

Specify the GitHub repository ('github_repo') you want to work with.
Extract the owner and repo name from the repository URL.
#### Define GitHub Token:

Set your GitHub token ('token') for authentication. Make sure it has the necessary permissions to interact with the repository.
#### Define Default Arguments:

Define default arguments for the DAG, including the owner, start date, and the number of retries in case of task failure.
#### Create the DAG Instance:

Create a DAG instance named 'Airflow-GitHub' with the previously defined default arguments.
Provide a description and set the schedule_interval to None (or manual) as it appears to be a trigger-based DAG.
#### Define PythonOperators for Each Function:

Create PythonOperator tasks for various functions used in the workflow.

- load_data_task: Calls the 'load_data' Python function.

- data_preprocessing_task: Calls the 'data_preprocessing' Python function and depends on 'load_data_task.'

- separate_data_outputs_task: Executes a custom function ('separate_data_outputs') to separate data outputs from 'data_preprocessing_task.' The output of 'data_preprocessing_task' is passed as an argument.

- build_save_model_task: Calls the 'build_model' Python function to build and save a model. It depends on 'separate_data_outputs_task.'

- load_model_task: Calls the 'load_model' Python function to load a saved model. It depends on 'build_save_model_task.'

#### Set Task Dependencies:

Define the task dependencies using the >> operator. Tasks are executed in sequence, ensuring that each task depends on the successful completion of its predecessor.
#### Command-Line Interaction:

If the script is run directly (not imported as a module), it allows command-line interaction with the DAG using the 'dag.cli()' method.




