import os
import cv2
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

images = []

def load_data():
    list_data = os.listdir("data")
    if len(list_data) > 0:
        for data in list_data:
            image = cv2.imread(data)
            images.append(image)
        return images
    else:
        return []

def preprocess_data():
    pass

def train_model():
    pass

def main():
    print(load_data())

if __name__ == "__main__":
    main()

