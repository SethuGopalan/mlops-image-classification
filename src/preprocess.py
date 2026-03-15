import tensorflow as tf

def load_datasets(train_path, test_path):
    train_data = tf.keras.utils.image_dataset_from_directory(train_path)
    test_data = tf.keras.utils.image_dataset_from_directory(test_path)
    return train_data, test_data

def main():
    train_path = "data/raw/training_set/training_set"
    test_path = "data/raw/test_set/test_set"

    train_data, test_data = load_datasets(train_path, test_path)

    print("Training dataset loaded.")
    print("Test dataset loaded.")
    print("Class names:", train_data.class_names)

if __name__ == "__main__":
    main()

