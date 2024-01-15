import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math

SAMPLES = 1500

# Set SEED
np.random.seed(786)
tf.random.set_seed(786)

# Uniform Random X-Values
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)

# View
plt.plot(x_values)
plt.show()

# Shuffle to prevent ordering
np.random.shuffle(x_values)

# Generate y-values
y_values = np.sin(x_values)

# Plot data
plt.plot(x_values,y_values, 'b.')
plt.show()

# Add random noise
y_values += 0.1*np.random.randn(*y_values.shape)

# Plot data
plt.plot(x_values,y_values, 'b.')
plt.show()

# Common practice to split data set - 60% training, 20% validation, 20% testing
TRAIN_SPLIT = int(0.6*SAMPLES)
TEST_SPLIT = int(0.2*SAMPLES + TRAIN_SPLIT)

x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

assert x_train.size + x_validate.size + x_test.size == SAMPLES

plt.plot(x_train, y_train, 'b.', label="Train")
plt.legend()
plt.show()

plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.legend()
plt.show()

plt.plot(x_test, y_test, 'r.', label="Test")
plt.legend()
plt.show()


##########################
# Setup Model
#########################

model = tf.keras.Sequential()

# First layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))

# Second layer
model.add(keras.layers.Dense(16, activation='relu',))

# Final layer
model.add(keras.layers.Dense(1))

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
training_info = model.fit(x_train, y_train, epochs=350, batch_size=64, validation_data=(x_validate, y_validate))

#######################################
# Metrics
######################################
# Draw loss graph
loss = training_info.history['loss']
validation_loss = training_info.history['val_loss']

epochs = range(1, len(loss) + 1)

SKIP = 50

plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], validation_loss[SKIP:], 'r.', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
# Draw Mean Absolute Error graph
mae = training_info.history['mae']
validation_mae = training_info.history['val_mae']

plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
plt.plot(epochs[SKIP:], validation_mae[SKIP:], 'r.', label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Convert to embedded-system model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
open("sinewave_model.tflite", 'wb').write(tflite_model)

# Generate C source/header
from tensorflow.lite.python.util import convert_bytes_to_c_source
source_text, header_text = convert_bytes_to_c_source(tflite_model, "sine_model", include_path="sine_model.h")
with open('sine_model.h', 'w') as file:
    file.write(header_text)
with open('sine_model.c', 'w') as file:
    file.write(source_text)
