# 一个线性回归的小实例
#   y = 1.477*X+0.089

import numpy as np
import matplotlib.pyplot as plt


# step1:采样数据
# 加上误差变量E   y = 1.477*x+0.089+E,E~N(0,0.01)
def load_data():
    dataset = []
    for i in range(100):
        x = np.random.uniform(-10, 10)

        # 采样高斯噪声
        eps = np.random.normal(0, 0.01)

        y = 1.477 * x + 0.089 + eps
        dataset.append([x, y])
    dataset = np.array(dataset)  # 转换为2D numpy数组
    return dataset


# step2：计算误差
def mse(b, w, points):
    # 根据当前的w，b参数计算均方差损失
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += ((w * x + b)-y) ** 2
    return totalError / float(len(points))


# step3:计算梯度
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对b的导数   grad_b = 2(wx+b-y)
        b_gradient += (2 / M) * ((w_current * x + b_current) - y)
        w_gradient += (2 / M) * x * ((w_current * x + b_current) - y)

    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


# step4:梯度更新
def gradient_descent(points, start_b, start_w, lr, num_iterations):
    loss_list = []
    epoch_list = []
    b = start_b
    w = start_w
    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        loss_list.append(loss)
        epoch_list.append(step)
        print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")

    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.plot(epoch_list, loss_list)
    plt.show()
    return [b, w]


if __name__ == '__main__':
    data = load_data()
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')

