# Import yaml so we can read the configuration file
import yaml

# Import MLflow for experiment tracking
import mlflow

# Import TensorFlow for building the neural network model
import tensorflow as tf

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


# This function builds the image classification model
def build_model():

    # Create data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Create the CNN model
    model = tf.keras.Sequential([

        # Define the input image shape
        tf.keras.Input(shape=(256, 256, 3)),

        # Normalize pixel values from 0-255 to 0-1
        tf.keras.layers.Rescaling(1.0 / 255),

        # Apply data augmentation during training
        data_augmentation,

        # First convolution block
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolution block
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third convolution block
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten feature maps into a vector
        tf.keras.layers.Flatten(),

        # Dense layer for learning high-level patterns
        tf.keras.layers.Dense(128, activation="relu"),

        # Dropout helps reduce overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer for binary classification
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Return the completed model
    return model


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

    # Build the model
    model = build_model()

    # Print confirmation that datasets and model are ready
    print("Training dataset is ready for the training pipeline.")
    print("Test dataset is ready for the training pipeline.")
    print("Class names:", train_data.class_names)
    print("Model architecture created successfully.")

    # Log a placeholder metric for now
    mlflow.log_metric("sample_accuracy", 0.0)

    # End the MLflow run cleanly
    mlflow.end_run()


# Run the main function when this file is executed directly
if __name__ == "__main__":
    main()
