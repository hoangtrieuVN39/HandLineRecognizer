import unittest
from data_preprocessing import *
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from constants import *

class TestMain(unittest.TestCase):
    # def test_load_data(self):
    #     print(len(load_data()))

    # def test_preprocess_data(self):
    #     print(preprocess_data())

    # def test_save_data(self):
    #     print(save_data())

    def test_main(self):
        main()

    def test_image_preprocessing(self):
        dataset_path = "data_processed"
        img_building = cv2.imread(os.path.join(dataset_path, 'MALE_0.jpg'))
        img_building = cv2.cvtColor(img_building, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    unittest.main(verbosity=2)