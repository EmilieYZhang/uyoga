import cv2
import tensorflow as tf

train_dir = "C:/Users/Emilie/OneDrive/Documents/UYOGAarchive/DATASET/TRAIN"
test_dir = "C:/Users/Emilie/OneDrive/Documents/UYOGAarchive/DATASET/TEST"

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255.,                                                                                                                   
                                                           rotation_range=0.2,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           zoom_range = 0.2, 
                                                           horizontal_flip=True)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255.)
train_data = train_gen.flow_from_directory(train_dir,
                                           target_size = (224,224),
                                           color_mode = "rgb",
                                           class_mode = "categorical",
                                           batch_size = 32
                                           )

test_data = test_gen.flow_from_directory(test_dir,
                                         target_size = (224,224),
                                         color_mode = "rgb",
                                         class_mode = "categorical",
                                         batch_size = 32
                                         )

labels = list(train_data.class_indices.keys())

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras import layers

model = tf.keras.Sequential([
                             layers.Conv2D(filters= 64, kernel_size= 2, activation= "relu", input_shape=(224,224,3)),
                             layers.MaxPooling2D(pool_size=2),

                             layers.Conv2D(filters= 64, kernel_size= 2, activation= "relu"),
                             layers.MaxPooling2D(pool_size= 2),

                             layers.Conv2D(filters= 64, kernel_size= 2, activation= "relu"),
                             layers.MaxPooling2D(pool_size= 2),

                             layers.Flatten(),

                             layers.Dense(512, activation= "relu"),

                             layers.Dense(512, activation = "relu"),

                             layers.Dense(512, activation = "relu"),

                             layers.Dense(5, activation="softmax")
])

model.summary()

model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

history = model.fit(train_data,
          epochs = 30,
          steps_per_epoch = len(train_data),
          validation_data = test_data,
          validation_steps = len(test_data)
          )

model_evaluation = model.evaluate(test_data)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(30)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print(f"Model Accuracy: {model_evaluation[1] * 100 : 0.2f} %")

model.save("UYOGA_MODEL")