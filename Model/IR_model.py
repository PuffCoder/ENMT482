import scipy as sp
import numpy as np
import matplotlib.pyplot as plt  # Import the necessary module
from scipy.stats import norm



### ********************************* IR SENSOR MODEL *******************************************
def ir1_model(x,a,b,c):
    return a/(x + b)  + c 

def ir2_model(x,a,b,c):
    return a/x**2 + b*x +c

def ir3_model(x,a0,a1,b,c):
    # return a/x+ b* x + c
    return a0/(x +a1)  + b * x + c

def ir33_model(x,a1,a2,b1,b2,c):
    return (a1 * x + b1)/(a2 * x + b2) + c
    return a1 / (x + a2) +  b1 * (x + b2) + c

def ir4_model(x,a1,b1,a2,b2,c1,c2,d1,d2,e1,e2,f):
    return (a1*(x+d1)**2 + b1*(x+e1) + c1) / (a2*(x+d2)**2 + b2*(x+e2) + c2) + f
    # return a1/x**3 + a2*x**3 + b1/x**2 + b2*x**2 + c1/x + c2*x + d
def sn1_model(x,a,b,c):
    return a*(x+b) + c



### ********************************* plot sensor and fit line *******************************************
def plot_fit_line(x,y,fit,str_title):
    fig, ax = plt.subplots(1)
    ax.set_title(str_title)
    ax.set_xlabel(r"Actual Distance (m) $x$")
    ax.set_ylabel(r"Voltage $(V)$")
    ax.plot(x, y, '.', alpha=1, label='Data')
    ax.plot(x, fit, label='Fit line')
    ax.legend()  # Add a legend
        



### ********************************* remove Outliers ********************************************
def remove_outlier(model, x, y,std_coeff=3):
    """
    Remove outliers from data and refit the model.
    
    Parameters:
        model (callable): The model function to fit.
        x (array-like): The x-values of the data.
        y (array-like): The y-values of the data.
    
    Returns:
        x_no_outlier (array): X-values after outlier removal.
        y_no_outlier (array): Y-values after outlier removal.
        filtered_fit (array): Fitted model values after outlier removal.
    """
    # Fit the initial model
    params, cov = sp.optimize.curve_fit(model, x, y)
    y_fit = model(x, *params)

    # Calculate the residuals
    residual = y - y_fit 
    
    # Mask out data points that are within 3 standard deviations
    mask_no_outlier = abs(residual) < np.std(residual) * std_coeff
    y_no_outlier = y[mask_no_outlier]
    x_no_outlier = x[mask_no_outlier]
 
    # Refit the model to the optimized data
    filtered_params, cov = sp.optimize.curve_fit(model, x_no_outlier, y_no_outlier)
    filtered_fit = model(x_no_outlier, *filtered_params)
 
    return x_no_outlier, y_no_outlier, filtered_fit, params







### ********************************* RESIDUAL & HISTOGRAM ********************************************
# def Residual_and_Histogram(x_no_outlier, y_no_outlier, filtered_fit):
#     """
#     Calculate the residuals and create a subplot with residual analysis.
    
#     Parameters:
#         x_no_outlier (array-like): X-values after outlier removal.
#         y_no_outlier (array-like): Y-values after outlier removal.
#         filtered_fit (array-like): Fitted model values after outlier removal.
#     """
#     # Calculate the residuals
#     residual = y_no_outlier - filtered_fit 
    
#     # Create subplots for residual analysis
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Specify figsize

#     # Plot the residuals
#     ax[0].plot(x_no_outlier, residual, '.', alpha=0.2)
#     ax[0].set_title("Residual")
#     ax[0].set_xlabel(r"Range $x$ (m)")
#     ax[0].set_ylabel(r"Measurement error $v$ (v)")

#     # Create a histogram of the residuals
#     ax[1].hist(residual, bins=40)
#     mu,std = norm.fit(residual)
    
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin,xmax,400)
#     p = norm.pdf(x,mu,std)
    
#     title_text = f'The mean {mu:.4e} & Var: { std*std:.4e})'  # 使用 f-string
    
#     ax[1].plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
    
    
    
#     ax[1].set_xlabel("Error $v$ (v)")
#     ax[1].set_ylabel("Density")
#     ax[1].set_title(title_text)
#     ax[1].grid()

#     plt.tight_layout()  # Adjust layout
#     plt.show()  # Display the plots
    
#     return mu, std**2 , residual



def calculate_ERR(x, y, model):
    """
    Calculate the residuals.
    Returns:
        array-like: Residuals.
    """
    return y - model(x)

def calculate_residual(x_no_outlier, y_no_outlier, filtered_fit):
    """
    Calculate the residuals.
    
    Parameters:
        x_no_outlier (array-like): X-values after outlier removal.
        y_no_outlier (array-like): Y-values after outlier removal.
        filtered_fit (array-like): Fitted model values after outlier removal.
    
    Returns:
        array-like: Residuals.
    """
    return y_no_outlier - filtered_fit


def plot_residual_analysis(x, residual):
    """
    Create a subplot with residual analysis.
    
    Parameters:
        x (array-like): X-values.
        residual (array-like): Residuals.
    """
    # Create subplots for residual analysis
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the residuals
    ax[0].plot(x, residual, '.', alpha=0.2)
    ax[0].set_title("Residual")
    ax[0].set_xlabel(r"Range $x$ (m)")
    ax[0].set_ylabel(r"Measurement error $v$ (v)")

    # Create a histogram of the residuals
    ax[1].hist(residual, bins=40, density=True,label='Histogram')
    mu, std = norm.fit(residual)
    
    xmin, xmax = plt.xlim()
    x_fit = np.linspace(xmin, xmax, 400)
    p = norm.pdf(x_fit, mu, std)
    
    title_text = f'Mean: {mu:.4e}, Variance: {std**2:.4e}'  # 使用 f-string
    
    ax[1].plot(x_fit, p, 'k', linewidth=2, label='Gaussian Fit')
    
    ax[1].set_xlabel("Error $v$ (v)")
    ax[1].set_ylabel("Density")
    ax[1].set_title(title_text)
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
    
    
def Residual_and_Histogram(x_no_outlier, y_no_outlier, filtered_fit):
    """
    Calculate the residuals and create a subplot with residual analysis.
    
    Parameters:
        x_no_outlier (array-like): X-values after outlier removal.
        y_no_outlier (array-like): Y-values after outlier removal.
        filtered_fit (array-like): Fitted model values after outlier removal.
    
    Returns:
        tuple: Mean, Variance, and Residuals.
    """
    residuals = calculate_residual(x_no_outlier, y_no_outlier, filtered_fit)
    plot_residual_analysis(x_no_outlier, residuals)
    mu, std = norm.fit(residuals)
    
    # return mu, std**2, residuals




def plot_sonar_data_and_model(x, v, sn1_x_no_outlier, sn1_v_no_outlier, model):
    """
    Create a plot to visualize the original data, filtered data, and fitted curve.
    
    Parameters:
        x (array-like): X-values (e.g., x measurements).
        sonar1 (array-like): Y-values (e.g., sonar sensor voltage readings).
        sn1_x_no_outlier (array-like): X-values after outlier removal.
        sn1_v_no_outlier (array-like): Y-values after outlier removal.
        sn1_model (function): Model function to generate the fitted curve.
        filtered_params (tuple): Parameters of the fitted curve.
    """
    # Create a plot to visualize the original data, filtered data, and fitted curve
    plt.figure(figsize=(12, 6))
    plt.plot(x, v, '.', alpha=0.2, label='Raw Data')
    plt.plot(sn1_x_no_outlier, sn1_v_no_outlier, '.', label='No Outlier Data')

    # Generate x values for the fitted curve
    x_values = np.linspace(0, 3.5, 100)
    
    # Calculate the fitted curve using the provided model and parameters
    fitted_curve = model(x_values)
    
    plt.plot(x_values, fitted_curve, label='Fitted Curve')

    # Add labels, legends, and title
    plt.xlabel('Distance (m)')
    plt.ylabel('Sensor Voltage (V)')
    plt.legend()
    plt.title('Sensor Data and Fitted Curve')
    # plt.xlim([0.5, 3.5])
    plt.ylim([0,4])
    # Display the plot
    plt.show()



# ****************************** Var ************************



def CreateVariancesLUT(x, err, step=0.1):
    if len(x) != len(err):
        raise ValueError("x和err的长度必须相同")

    variances = []
    x_segments = []
    start = 0
    end = step

    while end <= max(x):
        x_segment = [x_val for x_val in x if start <= x_val <= end]
        err_segment = [err_val for x_val, err_val in zip(x, err) if start <= x_val <= end]

        if len(x_segment) > 0:
            variance = np.var(err_segment)
            variances.append(variance)
            x_segments.append((start, end))


        start += step
        end += step

    return x_segments, variances


