import numpy as np
import matplotlib.pyplot as plt
import math


def linear_fit_with_uncertainty(x, y, y_uncertainty):
    N = len(x)

    # Calculate sums required for slope (m) and intercept (b)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)

    # Calculate the slope (m) and intercept (b)
    delta = N * sum_x2 - (sum_x)**2
    m = (N * sum_xy - sum_x * sum_y) / delta
    b = (sum_y * sum_x2 - sum_x * sum_xy) / delta

    # Calculate the variance of the y(x) values
    s2_yx = np.sum((y - (m*x + b))**2) / (N - 2)

    # Calculate uncertainties in the slope and intercept
    sm = np.sqrt(N * s2_yx / delta)
    sb = np.sqrt(s2_yx * sum_x2 / delta)

    return m, sm, b, sb

def calculate_chi_squared(x, y, y_uncertainty, x_uncertainty, m, b):
    # Calculate the expected y values
    y_expected = m*x + b

    # Combine uncertainties in x and y (simple linear propagation for slope*m*x)
    total_uncertainty = np.sqrt(y_uncertainty**2 + (m*x_uncertainty)**2)

    residuals = (y - y_expected) / total_uncertainty
    chi_squared = np.sum(residuals**2)

    # Degrees of freedom
    dof = len(y) - 2
    reduced_chi_squared = chi_squared / dof

    return chi_squared, reduced_chi_squared

def plot_fit_and_residuals(x, y, y_uncertainty, x_uncertainty, m, sm, b, sb):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8)
    )

    # First subplot: Data + fit
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = m*x_fit + b

    ax1.errorbar(x, y, xerr=x_uncertainty, yerr=y_uncertainty,
                 fmt='o', label='Data with error bars')
    ax1.plot(x_fit, y_fit, color='red',
             label=f'Fit: Time = ({m:.3f} ± {sm:.3f}) × LC + ({b:.3f} ± {sb:.3f})')

    ax1.set_xlabel('LC Unit')
    ax1.set_ylabel('Delay Time (µs)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Delay Time vs. LC Unit')

    # Second subplot: Residuals
    y_expected = m*x + b
    residuals = y - y_expected
    ax2.errorbar(x, residuals, xerr=x_uncertainty, yerr=y_uncertainty,
                 fmt='o', label='Residuals')
    ax2.axhline(0, color='red', linestyle='--', label='Zero residual')
    ax2.set_xlabel('LC Unit')
    ax2.set_ylabel('Residual (µs)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Residuals')

    plt.tight_layout()
    plt.show()


LC_units = np.array([
    1,   3,   5,   7,   9,
    11,  13,  15,  17,  19,
    21,  23,  25,  27,  29,
    31,  33,  35,  37,  39,
    41,
], dtype=float)

time_us = np.array([
     0.67,  6.83,  13.61, 21.05, 27.07,
    34.07, 40.05, 52.15, 58.09, 64.81,
    73.81, 80.68, 88.54, 94.01, 102.38,
    109.99,116.99,124.03,132.08,138.62,
    142.02,
], dtype=float)



# Uncertainties in LC (often negligible if LC is just an integer count?? Check)
LC_uncertainty = np.zeros_like(LC_units)

# Uncertainties in time measurements
time_uncertainty = np.full_like(time_us, 0.5) # Should be 0.005 instead? It gives really huge reduced chi-squared value tho...


m, sm, b, sb = linear_fit_with_uncertainty(LC_units, time_us, time_uncertainty)
plot_fit_and_residuals(LC_units, time_us, time_uncertainty, LC_uncertainty, m, sm, b, sb)

chi_squared, reduced_chi_squared = calculate_chi_squared(
    LC_units, time_us, time_uncertainty, LC_uncertainty, m, b
)

print(f"Slope (m): {m:.5f} ± {sm:.5f} µs/LC-unit")
print(f"Intercept (b): {b:.5f} ± {sb:.5f} µs")
print(f"Chi-squared: {chi_squared:.3f}")
print(f"Reduced Chi-squared: {reduced_chi_squared:.3f}")
