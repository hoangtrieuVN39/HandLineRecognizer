import unittest
from unittest.mock import patch
import os
import pandas as pd
from constants import *
from data_labeling import *

class TestMain(unittest.TestCase):
    def test_label_image_with_curves(self):
        new_points = label_image_with_curves(f"./data_processed/FEMALE_50.jpg")
        print(new_points)

    def test_show_points(self):
        show_points(f"./data_processed/FEMALE_100.jpg")

    def test_frame_points(self):
        points = [(158, 180), (146, 163), (136, 149), (120, 135), (101, 122), (108, 95), (116, 108), (129, 123), (144, 135), (165, 145), (115, 91), (125, 106), (141, 118), (161, 124), (182, 121)]
        frame_points(points, "FEMALE_50")
    
    @patch('data_labeling.label_image_with_curves')
    def test_main(self, mock_label_image_with_curves):
        # Mock the return value of label_image_with_curves
        mock_label_image_with_curves.return_value = [(158, 180), (146, 163), (136, 149), (120, 135), (101, 122), (108, 95), (116, 108), (129, 123), (144, 135), (165, 145), (115, 91), (125, 106), (141, 118), (161, 124), (182, 121)]
        
        main()  # Run the main function which will use the mocked function

        if os.path.exists(POINTS_PATH):
            df = pd.read_csv(POINTS_PATH)
        else:
            df = pd.DataFrame(columns=HEADER)
        
        print(df)

if __name__ == "__main__":
    unittest.main(verbosity=2)