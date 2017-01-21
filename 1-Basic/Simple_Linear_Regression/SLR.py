# -*- encoding=utf-8 -*-
# 简单现行回归：只有一个自变量 y=k*x+b 预测使 (y-y*)^2  最小
import numpy as np


def fitSLR(x, y):
    n = len(x)
    dinominator = 0
    numerator = 0
    for i in range(0, n):
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x)) ** 2

    print("numerator:" + str(numerator))
    print("dinominator:" + str(dinominator))

    b1 = numerator / float(dinominator)
    b0 = np.mean(y) - b1 * float(np.mean(x))
    return b0, b1


# y= b0+x*b1
def prefict(x, b0, b1):
    return b0 + x * b1


x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]

b0, b1 = fitSLR(x, y)
y_predict = prefict(6, b0, b1)
print("y_predict:" + str(y_predict))
