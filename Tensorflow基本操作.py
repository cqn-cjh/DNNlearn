import tensorflow as tf

#定义一个变量
x1=tf.Variable([1,2,3])
#定义一个常量
x2=tf.constant([4,5,6])
# 减法op
sub = tf.subtract(x1,x2)
# 加法op
add = tf.add(x1,sub)
print(sub)
print(add)
