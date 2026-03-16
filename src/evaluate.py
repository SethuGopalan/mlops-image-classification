# Import yaml so we can read the configuration file
import yaml

# Import pandas for saving evaluation results
import pandas as pd

# Import TensorFlow for loading and evaluating the saved model
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


# Main evaluation pipeline function
def main():

    # Load all project settings from the configuration file
    config = load_config()

    # Read test dataset path from config
    test_path = config["data"]["test_path"]

    # Read training dataset path from config
    train_path = config["data"]["train_path"]

    # Read image size from config
    image_size = config["training"]["image_size"]

    # Load datasets using the preprocessing pipeline
    train_data, test_data = load_datasets(train_path, test_path, image_size)

    # Load the saved trained model
    model = tf.keras.models.load_model("models/cats_dogs_model_v1.keras")

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(test_data)

    # Create a DataFrame to store evaluation results
    results_df = pd.DataFrame({
        "test_accuracy": [test_accuracy],
        "test_loss": [test_loss]
    })

    # Save evaluation results to the reports folder
    results_df.to_csv("reports/evaluation_results.csv", index=False)

    # Print evaluation results
    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)
    print("Evaluation results saved to reports/evaluation_results.csv")


# Run the main function when this file is executed directly
if __name__ == "__main__":
    main()
