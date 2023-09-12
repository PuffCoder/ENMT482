#!/usr/bin/env python3
"""My code for Part A"""

from matplotlib import RcParams
import numpy as np
import matplotlib.pyplot as plt
from ParamID import *


# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T




#### sonar 1 stuff
m, c = myIRLS1(time, sonar1)
yLS = m * time + c

plt.figure(figsize=(12, 6))


plt.subplot(121)
plt.plot(time, sonar1, '.', alpha=0.2)
plt.plot(time, yLS, '-')
# plt.plot(time, range_)
plt.title('Sonar1')
plt.xlabel('Time (s)')





#### infra 3 stuff
# m, c , sc= myLnLSv2(time[1000:len(time)], raw_ir3[1000:len(time)])
# plt.subplot(122)
# plt.plot(time, raw_ir3, '.', alpha=0.2)
# yLS2 = sc* np.log(time+m) + c
# print(f"m {m}, c {c}")
# plt.plot(time, yLS2, '-')

# m, c , o= my1LSv2(time, raw_ir3)
# # m, c, o = 5, 0.5, 1
# plt.subplot(122)
# plt.plot(time, raw_ir3, '.', alpha=0.2)
# yLS2 = m/(time+o) + c
# print(f"m {m}, c {c}, o {o}")
# plt.plot(time, yLS2, '-')


# m, c , o= myRootLS(time, raw_ir3)
# # m, c, o = 5, 0.5, 1
# plt.subplot(122)
# plt.plot(time, raw_ir3, '.', alpha=0.2)
# yLS2 = m * np.sqrt(time+o) + c
# print(f"m {m}, c {c}, o {o}")
# plt.plot(time, yLS2, '-')


a, b, c, d , e= myIRLS4(time, raw_ir3)
plt.subplot(122)
plt.plot(time, raw_ir3, '.', alpha=0.2)
yLS2 = a*time*time*time*time + b*time*time*time + c*time*time + d*time + e
plt.plot(time, yLS2, '-')



# a, m, c = myLS2(time, raw_ir3)
# plt.subplot(122)
# plt.plot(time, raw_ir3, '.', alpha=0.2)
# yLS2 = a*time*time + m*time + c
# plt.plot(time, yLS2, '-')



# m, c = myIRLS1(time[500:], raw_ir3[500:])
# plt.subplot(122)
# plt.plot(time, raw_ir3, '.', alpha=0.2)
# yLS2 = m*time + c
# plt.plot(time, yLS2, '-')


# plt.plot(time, range_)
plt.title('raw_ir3')
plt.xlabel('Time (s)')

# plt.show()




plt.figure(figsize=(6, 6))
y = raw_ir4

a, b, c, d , e= myIRLS4(time, y)
plt.plot(time, y, '.', alpha=0.2)
yLS2 = a*time*time*time*time + b*time*time*time + c*time*time + d*time + e
plt.plot(time, yLS2, '-') 


plt.show()







# # Plot sonar error
# plt.figure(figsize=(12, 5))

# plt.subplot(121)
# plt.plot(time, range_ - sonar1, '.', alpha=0.2)
# plt.axhline(0, color='k')
# plt.title('Sonar1 error')
# plt.xlabel('Time (s)')
# plt.ylabel('Error (m)')

# plt.subplot(122)
# plt.plot(time, range_ - sonar2, '.', alpha=0.2)
# plt.axhline(0, color='k')
# plt.title('Sonar2 error')
# plt.xlabel('Time (s)')


# # Plot IR sensor measurements
# plt.figure(figsize=(8, 7))

# plt.subplot(221)
# plt.plot(range_, raw_ir1, '.', alpha=0.5)
# plt.title('IR1')
# plt.ylabel('Measurement (V)')

# plt.subplot(222)
# plt.plot(range_, raw_ir2, '.', alpha=0.5)
# plt.title('IR2')

# plt.subplot(223)
# plt.plot(range_, raw_ir3, '.', alpha=0.5)
# plt.title('IR3')
# plt.xlabel('Range (m)')
# plt.ylabel('Measurement (V)')

# plt.subplot(224)
# plt.plot(range_, raw_ir4, '.', alpha=0.5)
# plt.title('IR4')
# plt.xlabel('Range (m)')



