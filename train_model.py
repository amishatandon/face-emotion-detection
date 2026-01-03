import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ----------------------------
# DEBUG: Confirm TensorFlow works
print("TensorFlow version:", tf.__version__)
# ----------------------------

# Image settings
IMG_SIZE = 48
BATCH_SIZE = 64

# Load images from folders
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# ----------------------------
# DEBUG: Check dataset folders
print("Train dataset folders:", os.listdir("dataset/train"))
print("Test dataset folders:", os.listdir("dataset/test"))
# ----------------------------

train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ----------------------------
# DEBUG: Check classes detected
print("Detected classes:", train_generator.class_indices)
print("Number of training images:", train_generator.samples)
print("Number of testing images:", test_generator.samples)
# ----------------------------

# Build CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# DEBUG: Confirm model summary
model.summary()
# ----------------------------

# Train model
print("Starting training...")
model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator
)

# Save model
model.save("emotion_model.h5")
print("✅ Model training complete. Saved as emotion_model.h5")
