import yaml
import mlflow

def load_config():
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    
    config = load_config()
    mlflow.start_run()
    
    mlflow.log_param("epochs", config["training"]["epochs"])
    mlflow.log_param("batch_size", config["training"]["batch_size"])
    mlflow.log_param("image_size", config["training"]["image_size"])
    
    print("Project:", config["project_name"])
    print("Epochs:", config["training"]["epochs"])
    print("Batch size:", config["training"]["batch_size"])
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
