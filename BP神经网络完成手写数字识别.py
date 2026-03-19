import matplotlib.pyplot as plt
import numpy as np

# 这是一个手写数据集
from sklearn.datasets import load_digits
# 用于标签二值化处理，把标签转化为one-hot编码
from sklearn.preprocessing import LabelBinarizer
# 用于拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
# 用于评估分类的结果
from sklearn.metrics import confusion_matrix,classification_report

# 定义sigmoid函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# 定义神经网络类
class NeuralNetwork:
    # 初始化网络，定义网络结构
    # 假设传入（64，100，10），说明定义：
    # 输入层64个神经元（图像规格是8*8像素），隐藏层100个，输出层10个（0-9十个数字）
    def __init__(self,layers):
        # 权值初始化，范围为-1～1
        self.W1 = np.random.random([layers[0],layers[1]])*2 - 1
        self.W2 = np.random.random([layers[1],layers[2]])*2 - 1
        # 初始化偏置值
        self.b1 = np.zeros([layers[1]])
        self.b2 = np.zeros([layers[2]])
        # 定义空list用于保存loss,accuracy
        self.loss = []
        self.accuracy = []

    # 训练模型
    # X为数据输入，T为数据标签，lr为学习率，steps为训练次数，batch为批次大小
    # 使用批量随机梯度下降法，每次随机抽取一个批次的数据进行训练
    def train(self,X,T,lr=0.1,steps=20000,test=5000,batch=50):
        # 进行steps+1次训练
        for i in range(steps+1):
            index = np.random.randint(0,X.shape[0],batch)  # 随机选择一个批次的数据
            x = X[index]
            L1 = sigmoid(np.dot(x,self.W1)+self.b1)
            L2 = sigmoid(np.dot(L1,self.W2)+self.b2)
            delta_L2 = (T[index]-L2)*sigmoid_derivative(L2)
            delta_L1 = delta_L2.dot(self.W2.T)*sigmoid_derivative(L1)
            self.W2 += lr*L1.T.dot(delta_L2) / x.shape[0]
            self.W1 += lr*x.T.dot(delta_L1) / x.shape[0]
            self.b2 = self.b2 + lr * np.mean(delta_L2,axis=0)
            self.b1 = self.b1 + lr * np.mean(delta_L1,axis=0)

            # 每训练5000次预测一次准确率
            if i % test == 0:
                Y2 = self.predict(X_test)
                predictions = np.argmax(Y2,axis=1)
                acc = np.mean(np.equal(predictions,y_test))
                l = np.mean(np.square(y_test - predictions)/2)
                self.accuracy.append(acc)
                self.loss.append(l)
                print(f'steps:{i} accuracy:{acc:.3f} loss:{l:.3f}')

    # 模型预测结果
    def predict(self,x):
        L1 = sigmoid(np.dot(x,self.W1)+self.b1)
        L2 = sigmoid(np.dot(L1,self.W2)+self.b2)
        return L2


steps = 30001
test  = 3000

# 载入数据
digits = load_digits()
# 得到数据
X = digits.data
# 得到标签
y = digits.target
#数据归一化，有助于加快训练速度
#X中原来的数值范围为0～255，归一化后变成0～1
X -= X.min()
X /= X.max() - X.min()

#分割数据，1/4为测试数据，3/4为训练数据
#有1347个数据，450个测试数据
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

nm = NeuralNetwork([64,100,10])
# 标签转化为one-hot编码
labels_train = LabelBinarizer().fit_transform(y_train)

print('training set')
nm.train(X_train,labels_train,test=test,steps=steps)

predictions = nm.predict(X_test)
predictions = np.argmax(predictions,axis=1)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

plt.plot(range(0,steps+1,test),nm.loss)
plt.xlabel('steps')
plt.ylabel('loss')
plt.show()

plt.plot(range(0,steps+1,test),nm.accuracy)
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.show()