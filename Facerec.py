import os

def collect_datasets():
    # Replace with the actual filename
    os.system("python Datasets_Collector.py")

def train_datasets():
    # Replace with the actual filename
    os.system("python TrainingDataset.py")

def open_face_recognition():
    # Replace with the actual filename
    os.system("python RecUpdated.py")

def main():
    while True:
        print("")
        print("----------------------------------------------------------")
        print("-----------Welcome to Face Recognition Model -------------")
        print("----------------------------------------------------------")
        print("")
        print("\nPlease select an option:")
        print("1. Collect the datasets")
        print("2. Train the datasets")
        print("3. Open face recognition")
        print("Press any other key to exit.")

        choice = input("Enter your choice: ")

        if choice == '1':
            collect_datasets()
        elif choice == '2':
            train_datasets()
        elif choice == '3':
            open_face_recognition()
        else:
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
