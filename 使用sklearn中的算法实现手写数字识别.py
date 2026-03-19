import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

digits = load_digits()
X = digits.data
y = digits.target

X -= X.min()
X /= X.max()-X.min()

x_train,x_test,y_train,y_tess = train_test_split(X,y,test_size=0.25,random_state=0)

# (100,20)代表隐藏层有2层，第一层100个神经元，第二层20个。
# 同理，（50）代表有一个隐藏层，50个神经元。
# max_iter为迭代次数，或者说训练次数
mlp = MLPClassifier(hidden_layer_sizes=(100,20),max_iter=1000,random_state=0)
# 使用fit函数来训练
mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
print(classification_report(y_tess,predictions))

# 封装的只是普通的BP神经网络，不具备深度学习算法