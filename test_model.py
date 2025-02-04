import unittest
import os
import cv2
import pandas as pd
import numpy as np
from model import *

class TestMain(unittest.TestCase):
    def test_get_model(self):
        model = MobileNetModel(lastFourTrainable=True)
        model.summary()

    def test_train_model(self):
        x = []
        y = []
        model = MobileNetModel(lastFourTrainable=True)

        if os.path.exists("points.csv"):
            df = pd.read_csv("points.csv")
        else:
            print("No points.csv file found")
            return

        for i in os.listdir(DATA_PROCESSED_PATH):
            if not i.endswith(".jpg"): continue
            img = cv2.imread(os.path.join(DATA_PROCESSED_PATH, i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMAGE_SIZE)
            img = np.stack((img,)*3, axis=-1)  # Convert 2D to 3D by stacking the grayscale image into 3 channels
            name = i.split(".")[0]

            if name in df['name'].values:
                x.append(img)
                y.append(df[df['name'] == name].iloc[0, 1:].values)

        # Convert x and y to numpy arrays
        x = np.array(x)
        y = np.array(y, dtype=np.float32)

        model.train(x, y)
        model.save("model.keras")

if __name__ == "__main__":
    unittest.main(verbosity=2)