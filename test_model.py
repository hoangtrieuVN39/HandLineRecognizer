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

        if os.path.exists("model.keras"):
            model.load("model.keras")

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
            img = np.stack((img,)*3, axis=-1)
            name = i.split(".")[0]

            if name in df['name'].values:
                x.append(img)
                y.append(df[df['name'] == name].iloc[0, 1:].values)

        # Convert x and y to numpy arrays
        x = np.array(x)
        y = np.array(y, dtype=np.float32)

        model.train(x, y)
        model.save("model.keras")

    def test_predict(self):
        if os.path.exists("points.csv"):
            df = pd.read_csv("points.csv")
        else:
            print("No points.csv file found")
            return
        model = MobileNetModel(lastFourTrainable=True)
        model.load("model.keras")
        img = cv2.imread(os.path.join(DATA_PROCESSED_PATH, "FEMALE_50.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack((img,)*3, axis=-1)
        prediction = model.predict(img.reshape(1, 224, 224, 3))
        print(prediction)
        print(df[df['name'] == "FEMALE_50"].iloc[0, 1:].values)
        for i in range(15):
            cv2.circle(img, (int(prediction[0, i*2]), int(prediction[0, i*2+1])), 3, (0, 255, 0), -1)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    unittest.main(verbosity=2)