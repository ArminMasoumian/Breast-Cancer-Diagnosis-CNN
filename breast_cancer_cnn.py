# Author: Armin Masoumian (masoumian.armin@gmail.com)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BreastCancerCNN:
    def __init__(self, img_shape=(50,50,3)):
        self.img_shape = img_shape
        self.model = self.build_model()

    def build_model(self):
        # Define the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, train_dir, val_dir, batch_size=32, epochs=10, data_augmentation=True):
        # Define data generators for data preprocessing and augmentation
        if data_augmentation:
            train_datagen = ImageDataGenerator(rescale=1./255,
                                               rotation_range=20,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               shear_range=0.1,
                                               zoom_range=0.1,
                                               horizontal_flip=True,
                                               fill_mode='nearest')
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
            
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Create data generators
        train_gen = train_datagen.flow_from_directory(train_dir,
                                                      target_size=self.img_shape[:2],
                                                      batch_size=batch_size,
                                                      class_mode='binary')
        
        val_gen = val_datagen.flow_from_directory(val_dir,
                                                  target_size=self.img_shape[:2],
                                                  batch_size=batch_size,
                                                  class_mode='binary')

        # Train the model
        history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen)

        return history

    def evaluate(self, test_dir):
        # Define data generator for test set
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(test_dir,
                                                    target_size=self.img_shape[:2],
                                                    batch_size=1,
                                                    shuffle=False,
                                                    class_mode='binary')

        # Evaluate the model
        loss, acc = self.model.evaluate(test_gen)

        return loss, acc

    def predict(self, X):
        # Make predictions using the model
        y_pred = self.model.predict(X)
        y_pred = np.round(y_pred)

        return y_pred
