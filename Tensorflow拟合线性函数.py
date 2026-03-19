import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD

x_data = np.random.random(100)
noise = np.random.normal(0,0.01,x_data.shape)

y_data = x_data*0.1+0.2 + noise
plt.scatter(x_data,y_data)
plt.show()

# 构建一个顺序模型
# 顺序模型为keras中的基本模型结构，就像汉堡一样，一层一层叠加网络
model = tf.keras.Sequential()

# Dense为全连接层
# 在模型中添加一个全连接层
# units为输出神经元个数，input_dim为输入神经元个数
model.add(tf.keras.layers.Dense(1, input_dim=1))

# 设置模型的优化器和代价函数，学习率为0.03
model.compile(optimizer=SGD(0.03),loss='mean_squared_error')

for step in range(2001):
    # 训练一个批次的数据，返回cost值
    cost = model.train_on_batch(x_data,y_data)
    if step % 500 == 0:
        print('cost:',cost)

# 使用predict对数据进行预测，得到预测值y_pred
y_pred = model.predict(x_data)
plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()