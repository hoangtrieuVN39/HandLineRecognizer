import tensorflow as tf

def inception_v3():
    model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    output = tf.keras.layers.GlobalAveragePooling2D()(model.get_layer("mixed10").output)
    new_model = tf.keras.Model(inputs=model.input, outputs=output)
    return new_model

def main():
    model = inception_v3()
    model.summary()
    print(model.output)

if __name__ == "__main__":
    main()
