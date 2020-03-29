#线性模型
# y=x*w
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x*w
def cost_f(xs,ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred-y)**2
        return cost/len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2*x*(x*w-y)
        return grad/len(xs)

print('Predict(before training...)', 4, forward(4))

grad_list = []
w_list = []
epoch_list = []
for epoch in range(100):
    cost_val = cost_f(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    grad_list.append(cost_val)
    w -= 0.1*grad_val
    w_list.append(w)
    epoch_list.append(epoch)
    print('Epoch:', epoch, 'w=',  w, 'loss=', cost_val)

print('Predict(after traing...)', 4, forward(4))
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.plot(epoch_list, grad_list)
plt.show()