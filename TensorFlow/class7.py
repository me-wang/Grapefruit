import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential

data = pd.read_csv('E:/pycharmWorkplace/Grapefruit/TensorFlow/data/Income.csv')


x = data.Education
y = data.Income


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.summary()

model.compile(optimizer='adam',
              loss='mse'
              )

history = model.fit(x, y, epochs=50000)


y1 = list(model.predict(x))
plt.scatter(x, y)
plt.plot(x, y1)
plt.show()

# <class 'list'>: [array([1.7790489], dtype=float32), array([1.8310765], dtype=float32), array([1.8883067], dtype=float32), array([1.9403344], dtype=float32), array([1.9923574], dtype=float32), array([2.0495923], dtype=float32), array([2.0446496], dtype=float32), array([2.1536474], dtype=float32), array([2.205675], dtype=float32), array([2.2629054], dtype=float32), array([2.3149328], dtype=float32), array([2.379924], dtype=float32), array([2.4244502], dtype=float32), array([2.4762182], dtype=float32), array([2.528246], dtype=float32), array([2.5854764], dtype=float32), array([2.6375039], dtype=float32), array([2.6895313], dtype=float32), array([2.7467618], dtype=float32), array([2.7987893], dtype=float32), array([2.850817], dtype=float32), array([2.9080472], dtype=float32), array([2.960075], dtype=float32), array([3.0121024], dtype=float32), array([3.0641298], dtype=float32), array([3.1213603], dtype=float32), array([3.173388], dtype=float32), array([3.2267118], dtype=float32), array([3.2826457], dtype=float32), array([3.3346734], dtype=float32)]