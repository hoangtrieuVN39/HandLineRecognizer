import os
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dropout, Dense, BatchNormalization, Conv2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tqdm

from constants import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileNetModel:
    def __init__(self, lastFourTrainable: bool = False):
        self.model = self.getModel(lastFourTrainable)
    
    def getModel(self, lastFourTrainable: bool = False) -> tf.keras.Model:
        try:
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
            )
            if lastFourTrainable:
                for layer in base_model.layers[:-4]:
                    layer.trainable = False
            else:
                base_model.trainable = False

            # Enhanced architecture
            x = base_model.output
            x = BatchNormalization()(x)
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            outputs = Dense(NUM_KEYPOINTS * 2)(x)

            model = Model(inputs=base_model.input, outputs=outputs)
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def train(self, 
        x: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> tf.keras.callbacks.History:
        try:
            # Normalize labels to [0,1] range
            # y = y / IMAGE_SIZE[0]  # Assuming square image

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
            self.model.compile(
                loss='mae',
                optimizer=optimizer,
                metrics=['mse']
            )

            x_train, x_val, y_train, y_val = train_test_split(
                x, y, 
                test_size=validation_split, 
                random_state=42
            )

            # Preprocess input images
            x_train = preprocess_input(x_train)
            x_val = preprocess_input(x_val)
            
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )

            # Enhanced callbacks
            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
            ]
            
            history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def save(self, path: str) -> None:
        try:
            # Save model with custom objects
            tf.keras.models.save_model(
                self.model,
                path,
                save_format='keras'
            )
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        try:
            # Load model with custom objects
            self.model = tf.keras.models.load_model(
                path,
            )
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(x)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def summary(self) -> None:
        return self.model.summary()
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        try:
            return self.model.evaluate(x, y)
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

def preprocess_image(image_path: str) -> np.ndarray:        
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MobileNetV2
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def load_data() -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(POINTS_PATH):
            raise FileNotFoundError(f"Points file not found at {POINTS_PATH}")
        
        df = pd.read_csv(POINTS_PATH)
        x: List[np.ndarray] = []
        y: List[np.ndarray] = []

        image_files = [f for f in os.listdir(DATA_PROCESSED_PATH) if f.endswith(".jpg")]
        
        # Use tqdm for progress tracking
        for image_file in tqdm.tqdm(image_files, desc="Loading dataset"):
            name = image_file.split(".")[0]
            if name not in df['name'].values:
                continue
                
            img_path = os.path.join(DATA_PROCESSED_PATH, image_file)
            img = preprocess_image(img_path)
            
            # Get keypoints and normalize to [0,1]
            label = df[df['name'] == name].iloc[0, 1:].values
            label = label.astype(np.float32)

            x.append(img)
            y.append(label)
        
        return np.array(x), np.array(y, dtype=np.float32)

def create_model():
    model = MobileNetModel(lastFourTrainable=True)
    model.summary()
    return model

def main():
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    model = create_model()
    # if os.path.exists(MODEL_PATH):
    #     logger.info("Loading existing model...")
    #     model.load(MODEL_PATH)
    
    x, y = load_data()
    logger.info(f"Dataset loaded: {len(x)} samples")
    
    model.train(x, y, epochs=100, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()