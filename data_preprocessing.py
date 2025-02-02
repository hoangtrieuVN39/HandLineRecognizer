import os
import cv2
import numpy as np
import pandas as pd

DATA_RAW_PATH = "data_raw"
GENDER = ['MALE', 'FEMALE']

def load_data():
    labels = []
    images = []
    if os.path.exists(DATA_RAW_PATH):
        for g in GENDER:
            list_data = os.listdir(f"{DATA_RAW_PATH}\{g}")[:50]
            if len(list_data) > 0:
                for data in list_data:
                    image_path = os.path.join(f"{DATA_RAW_PATH}\{g}", data)
                    image = cv2.imread(image_path)
                    if image is not None:
                        images.append(image)
                        labels.append(g)
                else:
                    print(f"Warning: Unable to read image {image_path}")
        return pd.DataFrame({"image": images, "label": labels})
    else:
        return []

def preprocess_data(data):
    for i in range(len(data)):
        data.iloc[i, 0] = cv2.resize(data.iloc[i, 0], (224, 224))
        data.iloc[i, 0] = cv2.cvtColor(data.iloc[i, 0], cv2.COLOR_BGR2GRAY)
    return data

def save_data(data, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(len(data)):
        cv2.imwrite(f"{dir}\{data.iloc[i, 1]}_{i}.jpg", data.iloc[i, 0])



def main():
    data = load_data()
    data = preprocess_data(data)
    
    # Show the first image with click coordinates
    if not data.empty:
        show_image_with_click_coordinates(data.iloc[0, 0])
    
    save_data(data, "data_processed")

if __name__ == "__main__":
    main()

