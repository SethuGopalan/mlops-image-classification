import yaml
import mlflow

def load_config():
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    
    config = load_config()
    mlflow.start_run()
    print("Project:", config["project_name"])
    print("Epochs:", config["training"]["epochs"])
    print("Batch size:", config["training"]["batch_size"])

if __name__ == "__main__":
    main()
