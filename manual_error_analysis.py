import numpy as np
import matplotlib.pyplot as plt


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


# Modified function to plot V vs Current and residuals on the same figure
def plot_fit_and_residuals(x, y, y_uncertainty, x_uncertainty, m, sm, b, sb):
    # Create figure with two subplots: one for the fit, one for the residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))

    # Plot the V vs. I with best-fit line on the first subplot
    current_fit_A = np.linspace(min(x), max(x), 100)
    voltage_fit_A = m * current_fit_A + b
    ax1.errorbar(x, y, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Data with error bars')
    ax1.plot(current_fit_A, voltage_fit_A, color='red', label=f'Fit: V = ({m:.2e} ± {sm:.2e})I + ({b:.2f} ± {sb:.2f})')
    ax1.set_xlabel('Current (A)')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Voltage vs. Current with Linear Fit')

    # Calculate residuals
    y_expected = m * x + b
    residuals = y - y_expected

    # Plot the residuals on the second subplot
    ax2.errorbar(x, residuals, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Residuals with error bars')
    ax2.axhline(0, color='red', linestyle='--', label='Zero residual')
    ax2.set_xlabel('Current (A)')
    ax2.set_ylabel('Residuals (V)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Residuals')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


# Example data (from the screenshot)
current = np.array([0.0291, 0.0138, 0.00237, 0.00006318])
voltage = np.array([6.341, 6.346, 6.356, 6.359])
voltage_uncertainty = np.array([0.0051705, 0.005173, 0.005178, 0.0051795])
current_uncertainty = np.array([0.1758, 0.06216, 0.03182, 0.00052296]) * 1e-3

# Apply the linear fit method again with current in A
m, sm, b, sb = linear_fit_with_uncertainty(current, voltage, voltage_uncertainty)

# Plot V vs Current with residuals below it, considering both current and voltage uncertainties
plot_fit_and_residuals(current, voltage, voltage_uncertainty, current_uncertainty, m, sm, b, sb)

# Calculate chi-squared and reduced chi-squared considering both current and voltage uncertainties
chi_squared, reduced_chi_squared = calculate_chi_squared(current, voltage, voltage_uncertainty, current_uncertainty, m,
                                                         b)

# Output the slope, intercept, uncertainties, chi-squared, and reduced chi-squared values
print(f"Slope (m): {m:.5e} ± {sm:.5e} V/A")
print(f"Intercept (b): {b:.5f} ± {sb:.5f} V")
print(f"Chi-squared: {chi_squared:.5f}")
print(f"Reduced Chi-squared: {reduced_chi_squared:.5f}")
