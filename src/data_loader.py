import os

def check_data_folder(path="data"):
    if not os.path.exists(path):
        print("Data folder not found.")
    else:
        print(f"Data folder '{path}' exists.")

if __name__ == "__main__":
    check_data_folder()
