import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

# 加载数据
iris = load_digits()
print('data shape:', iris.data.shape)
print('target shape:', iris.target.shape)
plt.imshow(iris.images[1], cmap='gray')
plt.show()
