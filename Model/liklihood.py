import numpy as np

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



def find_value_interval(value, intervals):
    """
    根据给定的值和区间，找到该值所在的区间索引。

    参数：
    - value：要查找区间的值。
    - intervals：包含区间的NumPy数组。假定区间是已排序的。

    返回：
    - interval_index：值所在的区间索引。如果值小于第一个区间下限，返回0。
      如果值大于最后一个区间上限，返回len(intervals) - 1。
    """
    if value < intervals[0]:
        return 0
    if value >= intervals[-1]:
        return len(intervals) - 1
    for i in range(len(intervals) - 1):
        if value >= intervals[i] and value < intervals[i + 1]:
            return i
    return -1  # 如果