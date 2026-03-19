import numpy as np
import matplotlib.pyplot as plt

# (4,3)
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1],
              [1,2,1]
              ])

# (4,1)
t = np.array([[1],
              [1],
              [-1],
              [-1]
              ])  #标签
W = np.random.random((3,1))
lr = 0.1

for i in range(100):
    print(f'epoch: {i}')
    print(W)
    Y = np.sign(np.dot(X, W))  #(4,1)
    E = t - Y
    dW = lr * np.dot(X.T,E)/X.shape[0]
    W = W + dW
    # if np.all(t==Y):
    #     break

k = -W[1]/W[2]
d = -W[0]/W[2]

#正样本的xy坐标（此xy坐标指的是坐标轴中的xy，与下面的x1y1和x2y2没关系）
x1=[4,3] #xy坐标为（4，3）
y1=[3,3]
#负样本的xy坐标
x2=[1,1]
y2=[2,1]

xdata = (0,5)
plt.plot(xdata,xdata * k + d,'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()

