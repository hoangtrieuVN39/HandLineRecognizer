import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from constants import *

class MobileNetModel:
    def __init__(self, lastFourTrainable=False):
        self.model = self.getModel(lastFourTrainable)

    def train(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        self.model.fit(
            datagen.flow(x_train, y_train, batch_size=32),
            epochs=50,
            validation_data=(x_val, y_val)
        )
    
    def getModel(self, lastFourTrainable=False):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = lastFourTrainable  # Freeze during initial training

        # Add regression head
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(NUM_KEYPOINTS * 2, activation='linear')(x)  # 2 coordinates per keypoint

        model = Model(inputs=base_model.input, outputs=outputs)

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
        
    def save(self, path):
        self.model.save(path)