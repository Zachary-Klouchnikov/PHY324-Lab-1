"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import curve_fit

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""
FUNCTIONS
"""
def linear_fit(x, m, b):
    """A linear fit function.

    Arguments:
    x -- independent variable
    m -- slope
    b -- y-intercept
    """

    return m * x + b

def quadratic_fit(x, a, b, c):
    """A quadratic fit function.

    Arguments:
    x -- independent variable
    a -- quadratic coefficient
    """

    return (a * (x ** 2)) + (b * x) + c

def quadratic_fit_error_propagation(x, x_err, a, a_err, b, b_err, c_err):
    """Calculates the uncertainty in a quadratic fit function.

    Arguments:
    x -- independent variable
    x_err -- uncertainty in independent variable
    a -- quadratic coefficient
    a_err -- uncertainty in quadratic coefficient
    b -- linear coefficient
    b_err -- uncertainty in linear coefficient
    c_err -- uncertainty in constant term
    """

    return np.sqrt(quadratic_fit_lite_error_propagation(x, x_err, a, a_err) ** 2 + ((b * x) * np.sqrt((x_err / x) ** 2 + (b_err / b) ** 2)) ** 2 + (c_err) ** 2)

def quadratic_fit_lite(x, a):
    """A quadratic fit function.

    Arguments:
    x -- independent variable
    a -- quadratic coefficient
    """

    return a * x ** 2

def quadratic_fit_lite_error_propagation(x, x_err, a, a_err):
    """Calculates the uncertainty in the quadratic fit lite function.

    Arguments:
    x -- independent variable
    x_err -- uncertainty in independent variable
    a -- quadratic coefficient
    a_err -- uncertainty in quadratic coefficient
    """

    return (a * x ** 2) * np.sqrt((2 * x_err / x) ** 2 + (a_err / a) ** 2)

def reduced_chi_square(observed, expected, uncertainties, num_params):
    """Calculates the reduced chi-square value of a data set.

    Arguments:
    observed -- array of observed values
    expected -- array of expected values
    uncertainties -- array of uncertainties in observed values
    num_params -- number of fitted parameters
    """

    chi_squared = np.sum(((observed - expected) ** 2) / (uncertainties ** 2))
    degrees_of_freedom = len(observed) - num_params

    return chi_squared / degrees_of_freedom

def wall_effect_correction(velocity, bead_radius, container_radius):
    """Returns the velocity with the wall effect correction.

    Arguments:
    velocity -- measured terminal velocity
    bead_radius -- radius of the bead
    container_radius -- radius of the tube
    """

    correction_factor = 1 / (1 - (2.104 * (bead_radius / container_radius)) + (2.089 * (bead_radius / container_radius) ** 3))

    return velocity * correction_factor

"""
ANALYSIS OF GLYCERINE DATA
"""
glycerine_data = {} # Dictionary to hold glycerine data

# Loading Glycerine Data
glycerine_path = []
for i in range(1, 6):
    for j in range(1, 6):
        path = Path(f'glycerine_data/bead_{i}_trial_{j}.txt')
        time, position = np.loadtxt(path, dtype = float, delimiter = '\t', skiprows = 2, unpack = True)
        glycerine_data[path.stem] = [time, position]
        glycerine_path.append(path.stem)

# Bead Diameter Measurements and Uncertainties
bead_diameters = {
    'bead_5': [6.31, 0.01], # mm
    'bead_4': [4.72, 0.01], # mm
    'bead_3': [3.12, 0.02], # mm
    'bead_2': [2.28, 0.06], # mm
    'bead_1': [1.52, 0.03]  # mm
}

"Analysis of Glycerin Data for Each Bead"
terminal_velocities = []
velocity_errors = []

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Trials for Bead 1
fig, axes = plt.subplots(nrows = 1, ncols = 2, gridspec_kw = {'wspace': 0}, figsize = (10, 5))

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []

i = 0
for key in glycerine_path[0 : 5]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Plotting data
    axes[0].errorbar(time, position, yerr = bead_diameters['bead_1'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i], label = f"{key}")

    # Filtering data for times between 30s and 50s
    filtered_indices = np.where((time >= 30) & (time <= 50))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position, sigma = bead_diameters['bead_1'][0] / 2)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    axes[1].errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_1'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i])
    axes[1].plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--', color = colors[i])

    i += 1

# Average fit line and rectangle to highlight fit region
axes[0].add_patch(plt.Rectangle((filtered_time[0], filtered_position[0]), np.abs(filtered_time[-1] - filtered_time[0]), np.abs(filtered_position[-1] - filtered_position[0]), linewidth = 3, edgecolor = 'black', facecolor = 'none'))

slope_error = np.sqrt(np.sum(np.array(slope_err) ** 2))
intercept_error = np.sqrt(np.sum(np.array(intercept_err) ** 2))

axes[1].plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', zorder = 5, label = f"Average Fit: y = ({np.mean(slopes):.2f}±{slope_error:.2f})t + ({np.mean(intercepts):.2f}±{intercept_error:.2f})")
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(slope_error)

# Limits
axes[0].set_xlim(time[0] - 1, time[-1] + 1)

axes[1].set_xlim(filtered_time[0] - 0.5, filtered_time[-1] + 0.5)

# Labels
fig.suptitle("Bead 1 Trials", fontsize = 12)
fig.supxlabel("Time (s)", fontsize = 12)
fig.supylabel("Position (mm)", fontsize = 12)

axes[1].yaxis.tick_right()

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

plt.tight_layout()
plt.savefig('figures\\bead 1 data.pdf')
plt.show()

# Trials for Bead 2
fig, axes = plt.subplots(nrows = 1, ncols = 2, gridspec_kw = {'wspace': 0}, figsize = (10, 5))

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []

i = 0
for key in glycerine_path[5 : 10]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Plotting data
    axes[0].errorbar(time, position, yerr = bead_diameters['bead_2'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i], label = f"{key}")

    # Filtering data for times between 15s and 50s
    filtered_indices = np.where((time >= 15) & (time <= 50))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position, sigma = bead_diameters['bead_2'][0] / 2)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    axes[1].errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_2'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i])
    axes[1].plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

    i += 1

# Average fit line and rectangle to highlight fit region
axes[0].add_patch(plt.Rectangle((filtered_time[0], filtered_position[0]), np.abs(filtered_time[-1] - filtered_time[0]), np.abs(filtered_position[-1] - filtered_position[0]), linewidth = 3, edgecolor = 'black', facecolor = 'none'))

slope_error = np.sqrt(np.sum(np.array(slope_err) ** 2))
intercept_error = np.sqrt(np.sum(np.array(intercept_err) ** 2))

axes[1].plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', zorder = 5, label = f"Average Fit: y = ({np.mean(slopes):.2f}±{slope_error:.2f})t + ({np.mean(intercepts):.2f}±{intercept_error:.2f})")
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(slope_error)

# Limits
axes[0].set_xlim(time[0] - 1, time[-1] + 4.5)

axes[1].set_xlim(filtered_time[0] - 1, filtered_time[-1] + 1.5)

# Labels
fig.suptitle("Bead 2 Trials", fontsize = 12)
fig.supxlabel("Time (s)", fontsize = 12)
fig.supylabel("Position (mm)", fontsize = 12)

axes[1].yaxis.tick_right()

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

plt.tight_layout()
plt.savefig('figures\\bead 2 data.pdf')
plt.show()

# Trials for Bead 3
fig, axes = plt.subplots(nrows = 1, ncols = 2, gridspec_kw = {'wspace': 0}, figsize = (10, 5))

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []

i = 0
for key in glycerine_path[10 : 15]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Plotting data
    axes[0].errorbar(time, position, yerr = bead_diameters['bead_3'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i], label = f"{key}")

    # Filtering data for times between 10s and 40s
    filtered_indices = np.where((time >= 10) & (time <= 40))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position, sigma = bead_diameters['bead_3'][0] / 2)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    axes[1].errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_3'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i])
    axes[1].plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

    i += 1

# Average fit line and rectangle to highlight fit region
axes[0].add_patch(plt.Rectangle((filtered_time[0], filtered_position[0]), np.abs(filtered_time[-1] - filtered_time[0]), np.abs(filtered_position[-1] - filtered_position[0]), linewidth = 3, edgecolor = 'black', facecolor = 'none'))

slope_error = np.sqrt(np.sum(np.array(slope_err) ** 2))
intercept_error = np.sqrt(np.sum(np.array(intercept_err) ** 2))

axes[1].plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', zorder = 5, label = f"Average Fit: y = ({np.mean(slopes):.2f}±{slope_error:.2f})t + ({np.mean(intercepts):.2f}±{intercept_error:.2f})")
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(slope_error)

# Limits
axes[0].set_xlim(time[0] - 1, time[-1] + 3)

axes[1].set_xlim(filtered_time[0] - 2, filtered_time[-1] + 1)

# Labels
fig.suptitle("Bead 3 Trials", fontsize = 12)
fig.supxlabel("Time (s)", fontsize = 12)
fig.supylabel("Position (mm)", fontsize = 12)

axes[1].yaxis.tick_right()

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

plt.tight_layout()
plt.savefig('figures\\bead 3 data.pdf')
plt.show()

# Trials for Bead 4
fig, axes = plt.subplots(nrows = 1, ncols = 2, gridspec_kw = {'wspace': 0}, figsize = (10, 5))

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []

i = 0
for key in glycerine_path[15 : 20]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Plotting data
    axes[0].errorbar(time, position, yerr = bead_diameters['bead_4'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i], label = f"{key}")

    # Filtering data for times between 5s and 17.5s
    filtered_indices = np.where((time >= 5) & (time <= 17.5))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position, sigma = bead_diameters['bead_4'][0] / 2)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    axes[1].errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_4'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i])
    axes[1].plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

    i += 1

# Average fit line and rectangle to highlight fit region
axes[0].add_patch(plt.Rectangle((filtered_time[0], filtered_position[0]), np.abs(filtered_time[-1] - filtered_time[0]), np.abs(filtered_position[-1] - filtered_position[0]), linewidth = 3, edgecolor = 'black', facecolor = 'none'))

slope_error = np.sqrt(np.sum(np.array(slope_err) ** 2))
intercept_error = np.sqrt(np.sum(np.array(intercept_err) ** 2))

axes[1].plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', zorder = 5, label = f"Average Fit: y = ({np.mean(slopes):.2f}±{slope_error:.2f})t + ({np.mean(intercepts):.2f}±{intercept_error:.2f})")
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(slope_error)

# Limits
axes[0].set_xlim(time[0] - 1, time[-1] + 1)

axes[1].set_xlim(filtered_time[0] - 0.5, filtered_time[-1] + 0.5)

# Labels
fig.suptitle("Bead 4 Trials", fontsize = 12)
fig.supxlabel("Time (s)", fontsize = 12)
fig.supylabel("Position (mm)", fontsize = 12)

axes[1].yaxis.tick_right()

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

plt.tight_layout()
plt.savefig('figures\\bead 4 data.pdf')
plt.show()

# Trials for Bead 5
fig, axes = plt.subplots(nrows = 1, ncols = 2, gridspec_kw = {'wspace': 0}, figsize = (10, 5))

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []

i = 0
for key in glycerine_path[20 : 25]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Plotting data
    axes[0].errorbar(time, position, yerr = bead_diameters['bead_5'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i], label = f"{key}")

    # Filtering data for times between 3s and 11s
    filtered_indices = np.where((time >= 3) & (time <= 11))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position, sigma = bead_diameters['bead_5'][0] / 2)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    axes[1].errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_5'][0] / 2, fmt = 'o', ms = 3.0, color = colors[i])
    axes[1].plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

    i += 1

# Average fit line and rectangle to highlight fit region
axes[0].add_patch(plt.Rectangle((filtered_time[0], filtered_position[0]), np.abs(filtered_time[-1] - filtered_time[0]), np.abs(filtered_position[-1] - filtered_position[0]), linewidth = 3, edgecolor = 'black', facecolor = 'none'))

slope_error = np.sqrt(np.sum(np.array(slope_err) ** 2))
intercept_error = np.sqrt(np.sum(np.array(intercept_err) ** 2))

axes[1].plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', zorder = 5, label = f"Average Fit: y = ({np.mean(slopes):.2f}±{slope_error:.2f})t + ({np.mean(intercepts):.2f}±{intercept_error:.2f})")
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(slope_error)

# Limits
axes[0].set_xlim(time[0] - 1, time[-1] + 1)

axes[1].set_xlim(filtered_time[0] - 0.5, filtered_time[-1] + 1)

# Labels
fig.suptitle("Bead 5 Trials", fontsize = 12)
fig.supxlabel("Time (s)", fontsize = 12)
fig.supylabel("Position (mm)", fontsize = 12)

axes[1].yaxis.tick_right()

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

plt.tight_layout()
plt.savefig('figures\\bead 5 data.pdf')
plt.show()

"Plotting Terminal Velocity as a Function of Bead Radius"
fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, gridspec_kw = {'hspace': 0, 'height_ratios': [3, 1, 1]}, figsize = (10, 7))

effective_error = np.sqrt(np.array(velocity_errors) ** 2 + (0.18) ** 2)

# Plotting terminal velocity as a function of bead radius
axes[0].errorbar(bead_diameters[f'bead_1'][0] / 2, terminal_velocities[0], yerr = effective_error[0], fmt = 'o', ms = 3.0, color = colors[0], label = 'Derived Terminal Velocities of Beads')
for i in range(2, 6):
    axes[0].errorbar(bead_diameters[f'bead_{i}'][0] / 2, terminal_velocities[i - 1], yerr = effective_error[i - 1], fmt = 'o', ms = 3.0, color = colors[i - 1])

# Fitting
popt, pcov = curve_fit(quadratic_fit_lite, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(terminal_velocities, quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), effective_error, 1)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), color = 'black', linestyle = '--', label = f"Theoretical Scaling: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[1].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities - quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), yerr = np.sqrt((np.array(effective_error)) ** 2 + (quadratic_fit_lite_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0])) ** 2), fmt = 'o', capsize = 2, color = 'black', label = 'Theoretical Scaling Residuals')
axes[1].axhline(0, color = 'gray', linestyle = '--')

popt, pcov = curve_fit(quadratic_fit, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(terminal_velocities, quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), effective_error, 3)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), color = 'teal', linestyle = '--', label = f"Quadratic Fit: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ + ({popt[1]:.2f}$\\pm${np.sqrt(np.diag(pcov))[1]:.2f})r + ({popt[2]:.2f}$\\pm${np.sqrt(np.diag(pcov))[2]:.2f}) ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[2].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities - quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), yerr = np.sqrt((np.array(effective_error)) ** 2 + (quadratic_fit_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2])) ** 2), fmt = 'o', capsize = 2, color = 'teal', label = 'Quadratic Fit Residuals')
axes[2].axhline(0, color = 'gray', linestyle = '--')

# Labels
fig.suptitle("Terminal Velocity vs. Bead Radius", fontsize = 12)
fig.supxlabel("Bead Radius (mm)", fontsize = 12)
fig.supylabel("Terminal Velocity (mm/s)", fontsize = 12)

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

axes[2].legend(fontsize = 12)
axes[2].grid()

plt.tight_layout()
plt.savefig('figures\\terminal velocity vs bead radius with systemic error.pdf')
plt.show()

corrected_terminal_velocities = [wall_effect_correction(terminal_velocities[i], bead_diameters[f'bead_{i + 1}'][0], 93.5) for i in range(5)]

"Plotting Terminal Velocity as a Function of Bead Radius With Wall Effect Correction"
fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, gridspec_kw = {'hspace': 0, 'height_ratios': [3, 1, 1]}, figsize = (10, 7))

effective_error = np.sqrt(np.array(velocity_errors) ** 2 + (0.11) ** 2)

# Plotting terminal velocity as a function of bead radius with wall effect correction
axes[0].errorbar(bead_diameters[f'bead_1'][0] / 2, terminal_velocities[0], yerr = effective_error[0], fmt = 'o', ms = 3.0, color = colors[0], label = 'Derived Terminal Velocities of Beads')
for i in range(2, 6):
    axes[0].errorbar(bead_diameters[f'bead_{i}'][0] / 2, corrected_terminal_velocities[i - 1], yerr = effective_error[i - 1], fmt = 'o', ms = 3.0, color = colors[i - 1])

# Fitting
popt, pcov = curve_fit(quadratic_fit_lite, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(corrected_terminal_velocities, quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), np.array(effective_error), 1)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), color = 'black', linestyle = '--', label = f"Theoretical Scaling: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[1].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities - quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), yerr = np.sqrt((np.array(effective_error)) ** 2 + (quadratic_fit_lite_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0])) ** 2), fmt = 'o', capsize = 2, color = 'black', label = 'Theoretical Scaling Residuals')
axes[1].axhline(0, color = 'gray', linestyle = '--')

popt, pcov = curve_fit(quadratic_fit, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(corrected_terminal_velocities, quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), np.array(effective_error), 3)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), color = 'teal', linestyle = '--', label = f"Quadratic Fit: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ + ({popt[1]:.2f}$\\pm${np.sqrt(np.diag(pcov))[1]:.2f})r + ({popt[2]:.2f}$\\pm${np.sqrt(np.diag(pcov))[2]:.2f}) ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[2].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities - quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), yerr = np.sqrt((np.array(effective_error)) ** 2 + (quadratic_fit_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2])) ** 2), fmt = 'o', capsize = 2, color = 'teal', label = 'Quadratic Fit Residuals')
axes[2].axhline(0, color = 'gray', linestyle = '--')

# Labels
fig.suptitle("Terminal Velocity vs. Bead Radius With Wall Effect Correction", fontsize = 12)
fig.supxlabel("Bead Radius (mm)", fontsize = 12)
fig.supylabel("Terminal Velocity (mm/s)", fontsize = 12)

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

axes[2].legend(fontsize = 12)
axes[2].grid()

plt.tight_layout()
plt.savefig('figures\\terminal velocity vs bead radius with wall effect correction and systematic error.pdf')
plt.show()

"Plotting Terminal Velocity as a Function of Bead Radius"
fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, gridspec_kw = {'hspace': 0, 'height_ratios': [3, 1, 1]}, figsize = (10, 7))

# Plotting terminal velocity as a function of bead radius
axes[0].errorbar(bead_diameters[f'bead_1'][0] / 2, terminal_velocities[0], yerr = effective_error[0], fmt = 'o', ms = 3.0, color = colors[0], label = 'Derived Terminal Velocities of Beads')
for i in range(2, 6):
    axes[0].errorbar(bead_diameters[f'bead_{i}'][0] / 2, terminal_velocities[i - 1], yerr = velocity_errors[i - 1], fmt = 'o', ms = 3.0, color = colors[i - 1])

# Fitting
popt, pcov = curve_fit(quadratic_fit_lite, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(terminal_velocities, quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), np.array(velocity_errors), 1)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), color = 'black', linestyle = '--', label = f"Theoretical Scaling: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[1].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities - quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), yerr = np.sqrt((np.array(velocity_errors)) ** 2 + (quadratic_fit_lite_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0])) ** 2), fmt = 'o', capsize = 2, color = 'black', label = 'Theoretical Scaling Residuals')
axes[1].axhline(0, color = 'gray', linestyle = '--')

popt, pcov = curve_fit(quadratic_fit, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(terminal_velocities, quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), np.array(velocity_errors), 3)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), color = 'teal', linestyle = '--', label = f"Quadratic Fit: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ + ({popt[1]:.2f}$\\pm${np.sqrt(np.diag(pcov))[1]:.2f})r + ({popt[2]:.2f}$\\pm${np.sqrt(np.diag(pcov))[2]:.2f}) ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[2].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], terminal_velocities - quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), yerr = np.sqrt((np.array(velocity_errors)) ** 2 + (quadratic_fit_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2])) ** 2), fmt = 'o', capsize = 2, color = 'teal', label = 'Quadratic Fit Residuals')
axes[2].axhline(0, color = 'gray', linestyle = '--')

# Labels
fig.suptitle("Terminal Velocity vs. Bead Radius Excluding Systematic Error", fontsize = 12)
fig.supxlabel("Bead Radius (mm)", fontsize = 12)
fig.supylabel("Terminal Velocity (mm/s)", fontsize = 12)

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

axes[2].legend(fontsize = 12)
axes[2].grid()

plt.tight_layout()
plt.savefig('figures\\terminal velocity vs bead radius.pdf')
plt.show()

corrected_terminal_velocities = [wall_effect_correction(terminal_velocities[i], bead_diameters[f'bead_{i + 1}'][0], 93.5) for i in range(5)]

"Plotting Terminal Velocity as a Function of Bead Radius With Wall Effect Correction"
fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, gridspec_kw = {'hspace': 0, 'height_ratios': [3, 1, 1]}, figsize = (10, 7))

# Plotting terminal velocity as a function of bead radius with wall effect correction
axes[0].errorbar(bead_diameters[f'bead_1'][0] / 2, terminal_velocities[0], yerr = effective_error[0], fmt = 'o', ms = 3.0, color = colors[0], label = 'Derived Terminal Velocities of Beads')
for i in range(2, 6):
    axes[0].errorbar(bead_diameters[f'bead_{i}'][0] / 2, corrected_terminal_velocities[i - 1], yerr = velocity_errors[i - 1], fmt = 'o', ms = 3.0, color = colors[i - 1])

# Fitting
popt, pcov = curve_fit(quadratic_fit_lite, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(corrected_terminal_velocities, quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), np.array(velocity_errors), 1)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), color = 'black', linestyle = '--', label = f"Theoretical Scaling: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[1].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities - quadratic_fit_lite(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0]), yerr = np.sqrt((np.array(velocity_errors)) ** 2 + (quadratic_fit_lite_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0])) ** 2), fmt = 'o', capsize = 2, color = 'black', label = 'Theoretical Scaling Residuals')
axes[1].axhline(0, color = 'gray', linestyle = '--')

popt, pcov = curve_fit(quadratic_fit, [bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities, sigma = velocity_errors)
chi2_reduced = reduced_chi_square(corrected_terminal_velocities, quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), np.array(velocity_errors), 3)

axes[0].plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), color = 'teal', linestyle = '--', label = f"Quadratic Fit: v$_{{term}}$ = ({popt[0]:.2f}$\\pm${np.sqrt(np.diag(pcov))[0]:.2f})r$^2$ + ({popt[1]:.2f}$\\pm${np.sqrt(np.diag(pcov))[1]:.2f})r + ({popt[2]:.2f}$\\pm${np.sqrt(np.diag(pcov))[2]:.2f}) ($\\chi_\\nu^2$ = {chi2_reduced:.2f})")
axes[2].errorbar([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], corrected_terminal_velocities - quadratic_fit(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), popt[0], popt[1], popt[2]), yerr = np.sqrt((np.array(velocity_errors)) ** 2 + (quadratic_fit_error_propagation(np.array([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)]), np.array([bead_diameters[f'bead_{i}'][1] / 2 for i in range(1, 6)]), popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2])) ** 2), fmt = 'o', capsize = 2, color = 'teal', label = 'Quadratic Fit Residuals')
axes[2].axhline(0, color = 'gray', linestyle = '--')

# Labels
fig.suptitle("Terminal Velocity vs. Bead Radius With Wall Effect Correction Excluding Systematic Error", fontsize = 12)
fig.supxlabel("Bead Radius (mm)", fontsize = 12)
fig.supylabel("Terminal Velocity (mm/s)", fontsize = 12)

axes[0].legend(fontsize = 12)
axes[0].grid()

axes[1].legend(fontsize = 12)
axes[1].grid()

axes[2].legend(fontsize = 12)
axes[2].grid()

plt.tight_layout()
plt.savefig('figures\\terminal velocity vs bead radius with wall effect correction.pdf')
plt.show()
