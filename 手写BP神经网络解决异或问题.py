import numpy as np
import matplotlib.pyplot as plt

# 神经网络结构为： 2-10-1
# 输入数据（4，2） 4个数据，每个数据2个特征
X = np.array([[1, 0],
              [1, 1],
              [0, 0],
              [0, 1],
              ])

# 标签（4，1）
T = np.array([[1],
              [0],
              [0],
              [1]])

# 参数矩阵W1(2,10) 其中2代表输入层数据特征数，10代表隐藏层神经元数量
W1 = np.array(np.random.random([2,10]))
# X.dot(W)结果是(4,10)

# 参数矩阵W2(10,1)  其中10代表隐藏层的信号总和，1代表输出层神经元数量
W2 = np.array(np.random.random([10,1]))

# 定义偏置值， 其中b1元素个数为10是因为隐藏层有10个神经元，需要添加偏置值b1 10次，b2同理
b1 =np.zeros([10])
b2 =np.zeros([1])

#定义学习率
lr = 0.1

#定义训练周期数量
epochs = 100001
#定义测试周期数量
test = 5000

loss = []

# 定义sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义sigmoid函数导数
def sigmoid_der(x):
    return x * (1 - x)

# 更新参数
def update():
    global W1, W2, b1, b2
    # 第一层（隐藏层信号总和）net1计算
    net1 = np.dot(X, W1)   #(4,10)
    Y1 = sigmoid(net1+b1)  #(4,10)
    # 第二层（输出层信号总和）net2计算
    net2 = np.dot(Y1, W2)  #(4,1)
    Y2 = sigmoid(net2+b2) #(4,1)

    # 第二层(输出层)学习信号计算&2 = （T-Y2）* f'(net2)
    dnet2 = (T - Y2) * sigmoid_der(Y2) / X.shape[0] #(4,1)

    # 第一层（隐藏层）学习信号计算&1 = &2.dot(W2.T) * f'(net1)
    dnet1 = dnet2.dot(W2.T) * sigmoid_der(Y1) / X.shape[0] #(4,10)

    # 参数矩阵变化量
    dW1 = lr * X.T.dot(dnet1)
    dW2 = lr * Y1.T.dot(dnet2)

    W1 = W1 + dW1
    W2 = W2 + dW2

    b1 = b1 + lr * np.mean(dnet1, axis=0)
    b2 = b2 + lr * np.mean(dnet2, axis=0)

def train():
    for i in range(epochs):
        update()
        if i % test == 0:
            # 隐藏层输出
            Y1 = sigmoid(np.dot(X, W1) + b1)
            # 输入层输出
            Y2 = sigmoid(np.dot(Y1, W2) + b2)
            # 计算loss值
            print(f'epochs:{i}  loss:{np.mean(np.square(T - Y2) / 2)}', )
            # 保存loss值
            loss.append(np.mean(np.square(T - Y2) / 2))


train()
#画训练周期数与loss的关系图
plt.plot(range(0,epochs,test),loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#隐藏层输出
Y1 = sigmoid(np.dot(X,W1)+b1)
#输出层输出
Y2 = sigmoid(np.dot(Y1,W2)+b2)
print('output:')
print(Y2)

#因为最终的分类只有0和1，所以我们把大于或者等于0.5的值归为1类，小于0.5的值归为0类
def predict(x):
    if x>0.5:
        return 1
    else:
        return 0

#map会根据提供的函数对制定序列做映射，相当于依次把L2中的值放到predict函数中计算，然后打印结果
print('predict:')
for i in Y2:
    print(predict(i))