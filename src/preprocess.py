# Import TensorFlow library for working with image datasets
import tensorflow as tf

# Import yaml so we can read the configuration file
import yaml

# This function loads the configuration settings from the config file
def load_config():

    # Open the configuration file located in configs/config.yaml
    with open("configs/config.yaml", "r") as file:

        # Convert the YAML content into a Python dictionary
        config = yaml.safe_load(file)

    # Return the configuration dictionary
    return config


# This function loads the training and test datasets
def load_datasets(train_path, test_path):

    # Load training images from the training directory
    train_data = tf.keras.utils.image_dataset_from_directory(train_path)

    # Load test images from the test directory
    test_data = tf.keras.utils.image_dataset_from_directory(test_path)

    # Return both datasets
    return train_data, test_data


# Main function that runs the preprocessing stage
def main():

    # Load configuration settings
    config = load_config()

    # Get training dataset path from the configuration file
    train_path = config["data"]["train_path"]

    # Get test dataset path from the configuration file
    test_path = config["data"]["test_path"]

    # Load datasets using the paths from the configuration
    train_data, test_data = load_datasets(train_path, test_path)

    # Print confirmation that training dataset loaded
    print("Training dataset loaded.")

    # Print confirmation that test dataset loaded
    print("Test dataset loaded.")

    # Print the detected class names
    print("Class names:", train_data.class_names)


# Run the main function when this file is executed
if __name__ == "__main__":
    main()


