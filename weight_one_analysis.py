import numpy as np
import matplotlib.pyplot as plt
import math

# Function to apply the step-by-step method from the screenshot
def linear_fit_with_uncertainty(x, y, y_uncertainty):
    N = len(x)

    # Calculate sums required for slope (m) and intercept (b)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Calculate the slope (m) and intercept (b)
    delta = N * sum_x2 - sum_x ** 2
    m = (N * sum_xy - sum_x * sum_y) / delta
    b = (sum_y * sum_x2 - sum_x * sum_xy) / delta

    # Calculate the variance of the y(x) values
    s2_yx = np.sum((y - (m * x + b)) ** 2) / (N - 2)

    # Calculate uncertainties in the slope and intercept
    sm = np.sqrt(N * s2_yx / delta)
    sb = np.sqrt(s2_yx * sum_x2 / delta)

    return m, sm, b, sb

# Function to calculate chi-squared for goodness of fit
def calculate_chi_squared(x, y, y_uncertainty, x_uncertainty, m, b):
    # Calculate the expected y values
    y_expected = m * x + b

    # Combine uncertainties in x and y
    total_uncertainty = np.sqrt(y_uncertainty ** 2 + (m * x_uncertainty) ** 2)

    residuals = (y - y_expected) / total_uncertainty
    chi_squared = np.sum(residuals ** 2)

    # Degrees of freedom
    dof = len(y) - 2
    reduced_chi_squared = chi_squared / dof

    return chi_squared, reduced_chi_squared

# Modified function to plot the transformed data and residuals on the same figure
def plot_fit_and_residuals(x, y, y_uncertainty, x_uncertainty, m, sm, b, sb):
    # Create figure with two subplots: one for the fit, one for the residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))

    # Plot the y vs. transformed x with best-fit line on the first subplot
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = m * x_fit + b
    ax1.errorbar(x, y, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Data with error bars')
    ax1.plot(x_fit, y_fit, color='red', label=f'Fit: y = ({m:.2e} ± {sm:.2e})x + ({b:.2f} ± {sb:.2f})')
    ax1.set_xlabel(r'$1/(\lambda - \lambda_0)$ (nm$^{-1}$)')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Relation between Scale Measuring y and $1/(\lambda - \lambda_0)$')

    # Calculate residuals
    y_expected = m * x + b
    residuals = y - y_expected

    # Plot the residuals on the second subplot
    ax2.errorbar(x, residuals, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Residuals with error bars')
    ax2.axhline(0, color='red', linestyle='--', label='Zero residual')
    ax2.set_xlabel(r'y')
    ax2.set_ylabel('Residuals')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Residuals')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

# Data provided
lambda_vals = np.array([447.1, 471.3, 492.2, 501.6, 587.6, 667.8, 410.2, 434.0, 486.1, 656.3])  # nm
lambda_uncertainty = 0.4  # nm
lambda0_uncertainty = 0.4 # nm
y_vals = np.array([14.12, 12.26, 11.51, 11.08, 8.50, 7.13, 17.62, 15.19, 11.84, 7.33])
y_uncertainty = 0.05

# Reference value for lambda_0
lambda_0 = 282.8  # nm

# Transform lambda to 1 / (lambda - lambda_0)
x_vals = 1 / (lambda_vals - lambda_0)

# Error propagation for transformed x
x_uncertainty = np.sqrt(2)*lambda_uncertainty / ((lambda_vals - lambda_0) ** 2)
print("x_uncertainty: ", x_uncertainty)

# Apply the linear fit method
m, sm, b, sb = linear_fit_with_uncertainty(x_vals, y_vals, y_uncertainty)

# Plot the transformed data with residuals below it
plot_fit_and_residuals(x_vals, y_vals, y_uncertainty, x_uncertainty, m, sm, b, sb)

# Calculate chi-squared and reduced chi-squared
chi_squared, reduced_chi_squared = calculate_chi_squared(x_vals, y_vals, y_uncertainty, x_uncertainty, m, b)

# Output the slope, intercept, uncertainties, chi-squared, and reduced chi-squared values
print(f"Slope (m): {m:.5e} ± {sm:.5e}")
print(f"Intercept (b): {b:.5f} ± {sb:.5f}")
print(f"Chi-squared: {chi_squared:.5f}")
print(f"Reduced Chi-squared: {reduced_chi_squared:.5f}")
