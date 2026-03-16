# Import yaml so we can read the configuration file
import yaml

# Import numpy for working with prediction values
import numpy as np

# Import TensorFlow for loading the saved model and image utilities
import tensorflow as tf


# This function loads the configuration settings from config.yaml
def load_config():

    # Open the configuration file in read mode
    with open("configs/config.yaml", "r") as file:

        # Convert YAML content into a Python dictionary
        config = yaml.safe_load(file)

    # Return the configuration dictionary
    return config


# Main prediction pipeline function
def main():

    # Load all project settings from the configuration file
    config = load_config()

    # Read image size from config
    image_size = config["training"]["image_size"]

    # Set the path of the image to predict
    image_path = "data/raw/test_set/test_set/cats/cat.4001.jpg"

    # Load the trained model from the models folder
    model = tf.keras.models.load_model("models/cats_dogs_model_v1.keras")

    # Load one image and resize it to match the training image size
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(image_size, image_size)
    )

    # Convert the image into an array
    image_array = tf.keras.utils.img_to_array(image)

    # Add one extra dimension so the model sees this as a batch
    image_array = np.expand_dims(image_array, axis=0)

    # Run prediction
    prediction = model.predict(image_array)

    # Print raw prediction value
    print("Raw prediction:", prediction[0][0])

    # Interpret binary classification result
    if prediction[0][0] > 0.5:
        print("Predicted class: dog")
    else:
        print("Predicted class: cat")


# Run the main function when this file is executed directly
if __name__ == "__main__":
    main()
