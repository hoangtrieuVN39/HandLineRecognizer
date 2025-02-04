import os
import pandas as pd
import tensorflow as tf

from constants import HEADER

def inception_v3():
    model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    output = tf.keras.layers.GlobalAveragePooling2D()(model.get_layer("mixed10").output)
    new_model = tf.keras.Model(inputs=model.input, outputs=output)
    return new_model

def train_model(df):
    model = inception_v3()
    model.compile(optimizer="adam", loss="mse")
    model.fit(df.iloc[:,1:].values, df.iloc[:,0].values, epochs=100)

def main():
    if os.path.exists("points.csv"):
        df = pd.read_csv("points.csv")
    else:
        df = pd.DataFrame(columns=HEADER)
    train_model(df)

if __name__ == "__main__":
    main()