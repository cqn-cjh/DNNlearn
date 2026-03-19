import tensorflow as tf
import time

# 创建一个简单的计算任务
print("测试 MPS 加速性能...")
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])

    start_time = time.time()
    c = tf.matmul(a, b)
    end_time = time.time()

    print(f"矩阵乘法完成时间: {end_time - start_time:.2f} 秒")
    print(f"计算结果形状: {c.shape}")