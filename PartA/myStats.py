#!/usr/bin/env python3
"""Stats stuff"""

from matplotlib import RcParams
import numpy as np
import matplotlib.pyplot as plt


N = lambda x, mean, var: np.exp(-0.5 *((x-mean)/var)**2)  / (var * np.sqrt(2*np.pi))


def VarError(x, error, xstep = 0.1):
    curx = xstep
    curbar = np.array([])

    mean_x = np.array([])
    var_x = np.array([])
    xaxis = np.array([])

    for i in range(0, len(x)):
        
        if (x[i] < curx):
            
            curbar= np.append(curbar, error[i])
        else:
            # print(curbar)
            error_var = np.var(curbar)
            error_mean = np.mean(curbar)
            # print(f"variance: {error_var},  mean: {error_mean}")
            mean_x = np.append(mean_x, error_mean)
            var_x = np.append(var_x, error_var)
            xaxis = np.append(xaxis, x[i])
            curbar = np.array([])
            curx += xstep
    
    return xaxis, mean_x, var_x
