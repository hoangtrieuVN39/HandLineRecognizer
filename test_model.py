import unittest
import os
import cv2
import pandas as pd
import numpy as np
from model import MobileNetModel, preprocess_image, preprocess_input
from constants import *
import logging
from typing import Tuple, List
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MobileNetModel(lastFourTrainable=True)
        if os.path.exists(MODEL_PATH):
            cls.model.load(MODEL_PATH)
            logger.info("Loaded existing model")
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if not os.path.exists(POINTS_PATH):
                raise FileNotFoundError(f"Points file not found at {POINTS_PATH}")
            
            df = pd.read_csv(POINTS_PATH)
            x: List[np.ndarray] = []
            y: List[np.ndarray] = []
            
            # Get list of image files first
            image_files = [f for f in os.listdir(DATA_PROCESSED_PATH) if f.endswith(".jpg")]
            
            # Use tqdm for progress tracking
            for image_file in tqdm(image_files, desc="Loading images"):
                name = image_file.split(".")[0]
                if name not in df['name'].values:
                    continue
                    
                # Load and preprocess image
                img_path = os.path.join(DATA_PROCESSED_PATH, image_file)
                img = preprocess_image(img_path)
                
                # Get corresponding labels
                label = df[df['name'] == name].iloc[0, 1:].values
                label = label/IMAGE_SIZE[0]

                x.append(img)
                y.append(label)
            
            return np.array(x), np.array(y, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack((img,)*3, axis=-1)
        return img
    

    def test_model_creation(self):
        try:
            model = MobileNetModel(lastFourTrainable=True)
            self.assertIsNotNone(model)
            model.summary()
            logger.info("Model creation test passed")
        except Exception as e:
            logger.error(f"Model creation test failed: {str(e)}")
            raise

    def test_model_training(self):
        try:
            # Load and preprocess data
            x, y = self.load_data()
            self.assertGreater(len(x), 0, "No training data loaded")
            
            # Train model
            history = self.model.train(
                x, y,
                epochs=100,
                batch_size=min(32, len(x)),  # Ensure batch size doesn't exceed dataset size
                validation_split=0.2
            )
            
            # Save modelq
            self.model.save(MODEL_PATH)
            logger.info("Model training test passed")
            
            # Basic assertions on training history
            self.assertIsNotNone(history)
            self.assertTrue(hasattr(history, 'history'))
            self.assertIn('loss', history.history)
            self.assertIn('val_loss', history.history)
            
        except Exception as e:
            logger.error(f"Model training test failed: {str(e)}")
            raise
    def test_model_prediction(self):
        """Test model prediction with proper preprocessing and visualization."""
        try:
            test_image_path = os.path.join(DATA_PROCESSED_PATH, "FEMALE_55.jpg")
            if not os.path.exists(test_image_path):
                raise FileNotFoundError(f"Test image not found at {test_image_path}")          

            # Load and preprocess test image
            img = preprocess_image(test_image_path)
            img_display = cv2.imread(test_image_path)  # Keep original for display
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            img_display = cv2.resize(img_display, IMAGE_SIZE)

            # Preprocess input for model
            img_input = preprocess_input(img.reshape(1, *IMAGE_SIZE, 3))
            
            # Load model and make prediction
            model = MobileNetModel(lastFourTrainable=True)
            model.load(MODEL_PATH)
            prediction = model.predict(img_input)
            
            # Verify prediction shape
            self.assertEqual(prediction.shape[1], NUM_KEYPOINTS * 2)
            
            # Load ground truth and normalize
            df = pd.read_csv(POINTS_PATH)
            ground_truth = df[df['name'] == "FEMALE_52"].iloc[0, 1:].values
            ground_truth = ground_truth.astype(np.float32)
            points = prediction[0]
            
            # Log predictions and ground truth
            logger.info("Predictions vs Ground Truth:")
            for i in range(0, len(points), 2):
                logger.info(f"Keypoint {i//2 + 1}:")
                logger.info(f"  Predicted: ({points[i]}, {points[i+1]})")
                logger.info(f"  Ground Truth: ({ground_truth[i]}, {ground_truth[i+1]})")
            
            # Visualize results
            self.visualize_keypoints(img_display, points, ground_truth)
            
        except Exception as e:
            logger.error(f"Model prediction test failed: {str(e)}")
            raise
            
    @staticmethod
    def visualize_keypoints(img: np.ndarray, predicted_points: List[int], ground_truth_points: List[int]) -> None:
        
        # Create a copy for visualization
        img_viz = img.copy()
        
        # Draw predicted keypoint connections
        # for i in range(3):
        #     pts = np.array(predicted_points[i*10:(i+1)*10], np.int32)
        #     pts = pts.reshape((-1, 1, 2))
        #     cv2.polylines(img_viz, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        pts = np.array(predicted_points[0:10], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_viz, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        pts = np.array(predicted_points[10:20], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_viz, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
        pts = np.array(predicted_points[20:30], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_viz, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        # # Draw ground truth keypoint connections
        # for i in range(3):
        #     pts = np.array(ground_truth_points[i*10:(i+1)*10], np.int32)
        #     pts = pts.reshape((-1, 1, 2))
        #     cv2.polylines(img_viz, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        
        # # Draw predicted keypoints
        # for i in range(0, len(predicted_points), 2):
        #     x, y = predicted_points[i], predicted_points[i+1]
        #     cv2.circle(img_viz, (x, y), 4, (0, 255, 0), -1)  # Green for predictions
            
        # # Draw ground truth keypoints
        # for i in range(0, len(ground_truth_points), 2):
        #     x, y = ground_truth_points[i], ground_truth_points[i+1]
        #     cv2.circle(img_viz, (x, y), 4, (255, 0, 0), -1)  # Red for ground truth

            
        # Display image
        
        cv2.imshow('Keypoints', img_viz)
        # cv2.resizeWindow('Keypoints', 1000, 1000)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    unittest.main(verbosity=2)