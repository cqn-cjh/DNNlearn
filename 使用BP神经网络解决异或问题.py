import numpy as np
import matplotlib.pyplot as plt


#输入数据(4,2)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
#标签
T = np.array([[0],
             [1],
             [1],
             [0]])
#定义一个2层神经网络：2-10-1
#输入层为2个神经元。隐藏层10个。
#输入层到隐藏层的参数矩阵W1
W1 = np.random.random([2,10])
#隐藏层到输出层的参数矩阵W2
W2 = np.random.random([10,1])
#学习率
lr = 0.1
#隐藏层偏置值
b1 = np.zeros([10])
#输出层偏置值
b2 = np.zeros([1])
#定义训练周期数量
epochs = 100001
#定义测试周期数量
test = 5000

#定义sigmoid函数与其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

#更新权值和偏置值
def update():
    global X,T,W1, W2, lr, b1, b2
    #隐藏层神经元信号计算
    L1 = sigmoid(np.dot(X,W1)+b1)    #(4,10)
    #输出层神经元信号计算
    L2 = sigmoid(np.dot(L1,W2)+b2)   #(4,1)
    #输出层的学习信号
    delta_L2 = (T-L2) * sigmoid_derivative(L2)  #(4,1)
    #隐藏层的学习信号
    delta_L1 = delta_L2.dot(W2.T) * sigmoid_derivative(L1)
    #求隐藏层到输出层的权值改变
    #由于一次计算了多个样本，所以需要求平均
    delta_W2 = lr * L1.T.dot(delta_L2) / X.shape[0]
    #输入层到隐藏层的权值改变
    #由于一次计算了多个样本，所以需要求平均
    delta_W1 = lr * X.T.dot(delta_L1) / X.shape[0]
    #更新权值
    W1 = W1 + delta_W1
    W2 = W2 + delta_W2
    #改变偏置值
    #由于一次计算了多个样本，所以需要求平均
    b1 = b1 + lr * np.mean(delta_L1,axis=0)
    b2 = b2 + lr * np.mean(delta_L2,axis=0)

#定义空list保存loss值
loss = []

#训练
for i in range(epochs):
    #更新权值
    update()
    #每训练5000次计算一次loss值
    if i % test == 0:
        #隐藏层输出
        L1 = sigmoid(np.dot(X,W1)+b1)
        #输入层输出
        L2 = sigmoid(np.dot(L1,W2)+b2)
        #计算loss值
        print(f'epochs:{i}  loss:{np.mean(np.square(T-L2)/2)}',)
        #保存loss值
        loss.append(np.mean(np.square(T-L2)/2))

#画训练周期数与loss的关系图
plt.plot(range(0,epochs,test),loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#隐藏层输出
L1 = sigmoid(np.dot(X,W1)+b1)
#输出层输出
L2 = sigmoid(np.dot(L1,W2)+b2)
print('output:')
print(L2)

#因为最终的分类只有0和1，所以我们把大于或者等于0.5的值归为1类，小于0.5的值归为0类
def predict(x):
    if x>0.5:
        return 1
    else:
        return 0

#map会根据提供的函数对制定序列做映射，相当于依次把L2中的值放到predict函数中计算，然后打印结果
print('predict:')
for i in L2:
    print(predict(i))
