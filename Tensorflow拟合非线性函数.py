import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD

# 首先我们尝试创建一个二次函数的散点图
x_data = np.linspace(0,1,200)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = np.square(x_data) + noise
# plt.scatter(x_data, y_data)
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=1,activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='tanh'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=SGD(0.1),loss='mean_squared_error')

for step in range(4001):
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost:',cost)

y_pred = model.predict(x_data)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred,'r-',lw=3)
plt.show()


# 为什么不能单独用relu呢？
# 因为对于relu（t），若t>0则还是线性函数，若t<0则导致relu（t）=0
# 而0导致反向传播时发生神经元死亡