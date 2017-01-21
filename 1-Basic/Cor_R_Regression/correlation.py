import math

import numpy as np


def compute_correlation(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    s_s_r = 0
    x_var = 0
    y_var = 0
    for i in range(0, len(X)):
        diff_x_x_bar = X[i] - x_bar
        diff_y_y_bar = Y[i] - y_bar
        s_s_r += diff_x_x_bar * diff_y_y_bar
        x_var += diff_x_x_bar ** 2
        y_var += diff_y_y_bar ** 2

    s_s_t = math.sqrt(x_var * y_var)
    return s_s_r / s_s_t


test_X = [1, 3, 8, 7, 9]
test_Y = [10, 12, 24, 21, 34]

cor = compute_correlation(test_X, test_Y)
print(cor)
print(cor ** 2)


def poly_fit(x, y, degree):
    results = {}
    co_effs = np.polyfit(x, y, degree)

    results['poly_normal'] = co_effs

    p = np.poly1d(co_effs)

    y_hat = p(x)
    y_bar = np.sum(y) / len(y)

    ss_reg = np.sum((y_hat - y_bar) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    results['determination'] = ss_reg / ss_tot

    return results


# noinspection PyTypeChecker
print(poly_fit(test_X, test_Y, 1)['poly_normal'])
# noinspection PyTypeChecker
print(poly_fit(test_X, test_Y, 1)['determination'])
