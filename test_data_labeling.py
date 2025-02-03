import unittest
from data_labeling import *
import os
import pandas as pd

class TestMain(unittest.TestCase):
    def test_label_image_with_curves(self):
        new_points = label_image_with_curves(f"./data_processed/FEMALE_50.jpg")
        print(new_points)

    def test_frame_points(self):
        points = [(158, 180), (146, 163), (136, 149), (120, 135), (101, 122), (108, 95), (116, 108), (129, 123), (144, 135), (165, 145), (115, 91), (125, 106), (141, 118), (161, 124), (182, 121)]
        frame_points(points, "FEMALE_50")
    
    def test_main(self):
        if os.path.exists("points.csv"):
            df = pd.read_csv("points.csv", names=HEADER)
        else:
            df = pd.DataFrame(columns=HEADER)
        for i in ['FEMALE_50', 'FEMALE_51']:
            new_points = label_image_with_curves(f"./data_processed/{i}.jpg")
            new_points = frame_points(new_points, f"{i}")
            df = pd.concat([df, new_points], ignore_index=True)
        df.to_csv("points.csv", index=False, header=False)
        print(df)

if __name__ == "__main__":
    unittest.main(verbosity=2)