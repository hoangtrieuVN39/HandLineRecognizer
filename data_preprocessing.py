import os
import cv2
import pandas as pd
from constants import *

def load_data():
    labels = []
    images = []
    if os.path.exists(DATA_RAW_PATH):
        for g in GENDER:
            list_data = os.listdir(f"{DATA_RAW_PATH}\{g}")
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
        data.iloc[i, 0] = cv2.resize(data.iloc[i, 0], IMAGE_SIZE)
        data.iloc[i, 0] = cv2.cvtColor(data.iloc[i, 0], cv2.COLOR_BGR2GRAY)
    return data

def save_data(data, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(len(data)):
        cv2.imwrite(f"{dir}\{name}.jpg", data.iloc[i, 0])

def main():

    if os.path.exists(DATA_RAW_PATH):
        for g in GENDER:
            i = 0
            list_data = os.listdir(f"{DATA_RAW_PATH}\{g}")
            if len(list_data) > 0:
                for data in list_data:
                    image_path = os.path.join(DATA_RAW_PATH, g, data)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = pd.DataFrame({"image": [image], "label": [g]})
                        image = preprocess_data(image)
                        save_data(image, DATA_PROCESSED_PATH, f"{g}_{i}")
                        i += 1
                else:
                    print(f"Warning: Unable to read image {image_path}")


    # data = load_data()
    # data = preprocess_data(data)
    
    # save_data(data, "data_processed")

if __name__ == "__main__":
    main()

