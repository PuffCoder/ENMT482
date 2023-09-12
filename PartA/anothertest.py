from matplotlib import RcParams
import numpy as np
import matplotlib.pyplot as plt
from ParamID import *


x = np.arange(10,100,5)
y = -np.log10(x+3) + 3
for i in range(0,len(x)):
    y[i] += (np.random.random()*2 -1) /100

m, c = myLogLS(x, y)


plt.figure(figsize=(12, 6))


plt.subplot(121)
plt.plot(x, y, '.', alpha=0.2)



yLS = -np.log10(x+m) + c


plt.plot(x, yLS, '-')
# plt.plot(time, range_)
plt.title('y')
plt.xlabel('x')

plt.show()
