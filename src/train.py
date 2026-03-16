# Import yaml so we can read the configuration file
import yaml

# Import MLflow for experiment tracking
import mlflow

# Import the dataset loading function from preprocess.py
from src.preprocess import load_datasets


# This function loads the configuration settings from config.yaml
def load_config():

    # Open the configuration file in read mode
    with open("configs/config.yaml", "r") as file:

        # Convert YAML content into a Python dictionary
        config = yaml.safe_load(file)

    # Return the configuration dictionary
    return config


# Main training pipeline function
def main():

    # Load all project settings from the configuration file
    config = load_config()

    # Read training dataset path from config
    train_path = config["data"]["train_path"]

    # Read test dataset path from config
    test_path = config["data"]["test_path"]

    # Read image size from config
    image_size = config["training"]["image_size"]

    # Start an MLflow run to track this training pipeline execution
    mlflow.start_run()

    # Add the project name as a tag in MLflow
    mlflow.set_tag("project_name", config["project_name"])

    # Log training parameters into MLflow
    mlflow.log_param("epochs", config["training"]["epochs"])
    mlflow.log_param("batch_size", config["training"]["batch_size"])
    mlflow.log_param("image_size", image_size)

    # Load training and test datasets using the preprocessing pipeline
    train_data, test_data = load_datasets(train_path, test_path, image_size)

    # Print confirmation that datasets are ready
    print("Training dataset is ready for the training pipeline.")
    print("Test dataset is ready for the training pipeline.")
    print("Class names:", train_data.class_names)

    # Log a placeholder metric for now
    mlflow.log_metric("sample_accuracy", 0.0)

    # End the MLflow run cleanly
    mlflow.end_run()


# Run the main function when this file is executed directly
if __name__ == "__main__":
    main()
