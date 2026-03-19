import numpy as np

#单神经元模拟

t = 1
lr = 1
X = np.array([[1, 0,-1]])  #样本矩阵，在这个代码中唯一
# 注意到，第一个1是偏置值的获取
W = np.array([[-5],
              [0],
              [0]
              ])

for i in range(10):
    print(f'epoch:{i+1}')
    print(W)
    y = np.sign(np.dot(X, W))
    dW =(t-y)*lr * X.T
    W = W + dW #更改之后的参数矩阵

