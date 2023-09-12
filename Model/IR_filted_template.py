import scipy as sp
import numpy as np
import matplotlib.pyplot as plt  # Import the necessary module



def ir1_model(x,a,b,c):
    return a/x**2  + b*x + c

# Fit the polynomial model to the data
ir1_polynomial_params, cov = sp.optimize.curve_fit(ir1_model, distance, raw_ir1)
ir1_polynomial_fit = ir1_model(distance, *ir1_polynomial_params)

### ********************************* Outliers ********************************************
# Calculate the residuals
residual_ir1_polynomial = raw_ir1 - ir1_polynomial_fit
mask_outlier = abs(residual_ir1_polynomial) < np.std(residual_ir1_polynomial) * 3
pruned_ir1 = raw_ir1[mask_outlier]
pruned_x = distance[mask_outlier]

### ********************************* Range ********************************************
mask_range = (distance >= 0.15) & (distance <= 1.5)
mask = mask_range & mask_outlier

ir1_x = distance[mask]
ir1_v = raw_ir1[mask]
ir1_filtered_params, cov = sp.optimize.curve_fit(ir1_model, ir1_x,ir1_v)
ir1_filtered_fit = ir1_model(ir1_x, *ir1_filtered_params)



# Create a figure and plot the original data and the polynomial fit
fig, ax = plt.subplots(1)
ax.plot(distance, raw_ir1, '.', alpha=1, label='Original Data')
ax.plot(ir1_x,ir1_v,'.',alpha=0.3, label='Pruned Data')
ax.plot(ir1_x,ir1_filtered_fit,'.',alpha = 0.2, label="in range" )
ax.plot(distance, ir1_polynomial_fit,color = 'black', label='Polynomial Fit')
ax.legend()  # Add a legend

ax.plot()






### ********************************* remove Outliers ********************************************
ir1_pruned_params, cov = sp.optimize.curve_fit(ir1_model, ir1_x, ir1_v)
ir1_pruned_fit = ir1_model(ir1_x, *ir1_pruned_params)
residual_without_Outlier_ir1 = ir1_pruned_fit - ir1_v

# *********************** .  ***********************
# Create subplots for residual analysis
fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Specify figsize

# Plot the residuals
ax[0].plot(ir1_x, residual_without_Outlier_ir1, '.', alpha=0.2)
ax[0].set_xlabel(r"Range $x$")
ax[0].set_ylabel(r"Residuals")

# Create a histogram of the residuals
ax[1].hist(residual_without_Outlier_ir1, bins=20, density=True)
ax[1].set_xlabel("Residual Value")
ax[1].set_ylabel("Density")
ax[1].grid()

plt.tight_layout()  # Adjust layout
plt.show()  # Display the plots


# def model_nonlinear_least_squares_fit(r,v,iterations=100):
#     N = len(r)
#     A = np.ones((N,3))
#     k = np.zeros(3)
    
#     for i in range(iterations):
#         for n in range(N):
#             A[n,1] = 1 / (r[n] + k[2]) 
#             A[n,2] = -k[1] / (r[n] + k[2]) ** 2
            
#         deltak, res, rank, s = lstsq(A,v - model(r,k),rcond = -1)
#         k += deltak
#     return k


# filename = 'partA/calibration.csv'
# data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
# index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
#     sonar1, sonar2 = data.T

# params, cov = sp.optimize.curve_fit(model_log, distance, raw_ir3)
# y_fit = model_log(distance, *params