import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import functions

data = np.loadtxt("analysis/aanalysis set 2/p2.3.txt", skiprows=2)

# Extract x and y positions
x_pixels = data[:, 0]
y_pixels = data[:, 1]

# Constants and Conversion Factors
conversion_factor = 0.1155  # microns per pixel
T = 296.5  # Temperature in Kelvin
viscosity =1.00e-3 # Viscosity in SI units (Pa.s)
radius = 1.9e-6 / 2  # Radius of the bead in meters
k_B = 1.38e-23  # Boltzmann constant

# Convert pixel data to microns
x_microns = x_pixels * conversion_factor
y_microns = y_pixels * conversion_factor

# Calculate the displacement (distance traveled) at each time step
displacement = np.sqrt(np.diff(x_microns)**2 + np.diff(y_microns)**2)

# Time between frames
time_intervals = np.arange(1, len(displacement) + 1) * 0.5

# Mean Squared Displacement
mean_squared_displacement = np.cumsum(displacement**2)


# Uncertainties
position_uncertainty = 0.1
time_uncertainty = 0.03
viscosity_uncertainty = 0.05e-3
radius_uncertainty = 0.1e-6 / 2
T_uncertainty = 0.5

# Uncertainty
displacement_uncertainty = np.sqrt(2) * position_uncertainty  # Uncertainty for each displacement
msd_uncertainty = 2 * displacement * displacement_uncertainty

# Find best fit
m, sm, b, sb = functions.linear_fit_with_uncertainty(time_intervals, mean_squared_displacement)
functions.plot_fit_and_residuals(time_intervals, mean_squared_displacement, msd_uncertainty, time_uncertainty, m, sm, b, sb)
print(f"Slope (m): {m:.5e} ± {sm:.5e} ")
print(f"Intercept (b): {b:.5f} ± {sb:.5f} V")

# Find D and k
D = (m/4) * 1e-12
stokes_drag = 6*np.pi*viscosity*radius
k_calculated = D*stokes_drag/T
D_uncertainty = (sm / 4) * 1e-12

# Error propagation for k uncertainty
k_calculated_uncertainty = np.sqrt(
    (6 * np.pi * viscosity * radius / T * D_uncertainty) ** 2 +  # Uncertainty from D
    (6 * np.pi * D * radius / T * viscosity_uncertainty) ** 2 +  # Uncertainty from viscosity
    (6 * np.pi * D * viscosity / T * radius_uncertainty) ** 2 +  # Uncertainty from radius
    (6 * np.pi * D * viscosity * radius / T**2 * T_uncertainty) ** 2  # Uncertainty from temperature
)

print(f"Calculated Boltzmann Constant (k): {k_calculated:.4e} J/K ± {k_calculated_uncertainty:.4e}")

# Compare to accepted value of k
k_accepted = 1.38e-23
percent_difference = np.abs((k_calculated - k_accepted) / k_accepted) * 100
print(f"Percent Difference compared to accepted value of k: {percent_difference:.4e} %")