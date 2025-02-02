import unittest
from data_preprocessing import *
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

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
        
        
        # image = pd.read_csv("data_processed/data_processed.csv")
        # # image = np.array(image["image"])
        # # image = image[0]
        # # print(image.shape)
        # print(image)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)

    def test_show_image_with_click_coordinates(self):
        # Load a sample image from the processed data
        full_coordinates = []
        dataset_path = "data_processed"
        image_paths = os.listdir(dataset_path)
        for image_path in image_paths:
            if not image_path.endswith(".jpg"): continue
            while True:
                coordinates = []
                image = cv2.imread(os.path.join(dataset_path, image_path))
                def click_event(event, x, y, flags, param)q:
                    if event == cv2.EVENT_LBUTTONDOWN:
                        coordinates.append((x, y))
                        cv2.imshow('Image', image)

                cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Image', 800, 600)
                
                # Get screen resolution
                screen_width = 1920  # Replace with your screen width
                screen_height = 1080  # Replace with your screen height
                
                # Calculate position to center the window
                x_pos = (screen_width - 800) // 2
                y_pos = (screen_height - 600) // 2
                
                # Move the window to the center
                cv2.moveWindow('Image', x_pos, y_pos)
                
                cv2.imshow('Image', image)
                cv2.setMouseCallback('Image', click_event)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                full_coordinates.append(coordinates)

if __name__ == "__main__":
    unittest.main(verbosity=2)