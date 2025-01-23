import unittest
from main import *

class TestMain(unittest.TestCase):
    def load_data(self):
        self.assertEqual(load_data(), ["image.png"])

if __name__ == "__main__":
    unittest.main()
