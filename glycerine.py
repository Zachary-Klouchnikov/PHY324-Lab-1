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

"Plotting Time vs. Position for Glycerine Data"
fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (6, 10))

# Plotting data for bead 1
for key in glycerine_path[0 : 5]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[0].errorbar(time, position, yerr = bead_diameters['bead_1'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')

# Labels
axes[0].set_title('Bead 1 Trials', fontsize = 12)

axes[0].legend(fontsize = 12)
axes[0].grid()

# Plotting data for bead 2
for key in glycerine_path[5 : 10]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[1].errorbar(time, position, yerr = bead_diameters['bead_2'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')

# Labels
axes[1].set_title('Bead 2 Trials', fontsize = 12)

axes[1].legend(fontsize = 12)
axes[1].grid()

# Plotting data for bead 3
for key in glycerine_path[10 : 15]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[2].errorbar(time, position, yerr = bead_diameters['bead_3'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')

# Labels
axes[2].set_title('Bead 3 Trials', fontsize = 12)
axes[2].set_ylabel('Position (mm)', fontsize = 12)

axes[2].legend(fontsize = 12)
axes[2].grid()

# Plotting data for bead 4
for key in glycerine_path[15 : 20]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[3].errorbar(time, position, yerr = bead_diameters['bead_4'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')

# Labels
axes[3].set_title('Bead 4 Trials', fontsize = 12)

axes[3].legend(fontsize = 12)
axes[3].grid()

# Plotting data for bead 5
for key in glycerine_path[20 : 25]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[4].errorbar(time, position, yerr = bead_diameters['bead_5'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')

# Labels
axes[4].set_title('Bead 5 Trials', fontsize = 12)
axes[4].set_xlabel('Time (s)', fontsize = 12)

axes[4].legend(fontsize = 12)
axes[4].grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.tight_layout()
plt.show()

"Analysis of Glycerin Data for Each Bead"
terminal_velocities = []
velocity_errors = []

# Trials for Bead 1
plt.figure()

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []
for key in glycerine_path[0 : 5]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 30s and 50s
    filtered_indices = np.where((time >= 30) & (time <= 50))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position, sigma = bead_diameters['bead_1'][1] / 2)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    plt.errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_1'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}±{np.mean(slope_err):.2f}x + {np.mean(intercepts):.2f}±{np.mean(intercept_err):.2f}')
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(np.mean(slope_err))

# Labels
plt.title('Bead 1 Trials', fontsize = 12)
plt.xlabel('Time (s)', fontsize = 12)
plt.ylabel('Position (mm)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()

# Trials for Bead 2
plt.figure()

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []
for key in glycerine_path[5 : 10]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 15s and 50s
    filtered_indices = np.where((time >= 15) & (time <= 50))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    plt.errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_2'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}±{np.mean(slope_err):.2f}x + {np.mean(intercepts):.2f}±{np.mean(intercept_err):.2f}')
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(np.mean(slope_err))

# Labels
plt.title('Bead 2 Trials', fontsize = 12)
plt.xlabel('Time (s)', fontsize = 12)
plt.ylabel('Position (mm)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()

# Trials for Bead 3
plt.figure()

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []
for key in glycerine_path[10 : 15]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 10s and 40s
    filtered_indices = np.where((time >= 10) & (time <= 40))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    plt.errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_3'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}±{np.mean(slope_err):.2f}x + {np.mean(intercepts):.2f}±{np.mean(intercept_err):.2f}')
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(np.mean(slope_err))

# Labels
plt.title('Bead 3 Trials', fontsize = 12)
plt.xlabel('Time (s)', fontsize = 12)
plt.ylabel('Position (mm)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()

# Trials for Bead 4
plt.figure()

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []
for key in glycerine_path[15 : 20]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 5s and 17.5s
    filtered_indices = np.where((time >= 5) & (time <= 17.5))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    plt.errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_4'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}±{np.mean(slope_err):.2f}x + {np.mean(intercepts):.2f}±{np.mean(intercept_err):.2f}')
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(np.mean(slope_err))

# Labels
plt.title('Bead 4 Trials', fontsize = 12)
plt.xlabel('Time (s)', fontsize = 12)
plt.ylabel('Position (mm)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()

# Trials for Bead 5
plt.figure()

# Analysing and plotting data for each trial
slopes = []
slope_err = []
intercepts = []
intercept_err = []
for key in glycerine_path[20 : 25]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 3s and 11s
    filtered_indices = np.where((time >= 3) & (time <= 11))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function and storing parameters
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    slope_err.append(np.sqrt(np.diag(pcov))[0])
    intercepts.append(popt[1])
    intercept_err.append(np.sqrt(np.diag(pcov))[1])

    plt.errorbar(filtered_time, filtered_position, yerr = bead_diameters['bead_5'][1] / 2, fmt = 'o', ms = 3.0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}±{np.mean(slope_err):.2f}x + {np.mean(intercepts):.2f}±{np.mean(intercept_err):.2f}')
terminal_velocities.append(np.mean(slopes))
velocity_errors.append(np.mean(slope_err))

# Labels
plt.title('Bead 5 Trials', fontsize = 12)
plt.xlabel('Time (s)', fontsize = 12)
plt.ylabel('Position (mm)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()

"Plotting Terminal Velocity as a Function of Bead Radius"
plt.figure()

# Plotting terminal velocity as a function of bead radius
plt.errorbar(bead_diameters['bead_1'][0] / 2, terminal_velocities[0], xerr = bead_diameters['bead_1'][1] / 2, yerr = velocity_errors[0], fmt = 'o', ms = 3.0, label = f'Bead 1')
plt.errorbar(bead_diameters['bead_2'][0] / 2, terminal_velocities[1], xerr = bead_diameters['bead_2'][1] / 2, yerr = velocity_errors[1], fmt = 'o', ms = 3.0, label = f'Bead 2')
plt.errorbar(bead_diameters['bead_3'][0] / 2, terminal_velocities[2], xerr = bead_diameters['bead_3'][1] / 2, yerr = velocity_errors[2], fmt = 'o', ms = 3.0, label = f'Bead 3')
plt.errorbar(bead_diameters['bead_4'][0] / 2, terminal_velocities[3], xerr = bead_diameters['bead_4'][1] / 2, yerr = velocity_errors[3], fmt = 'o', ms = 3.0, label = f'Bead 4')
plt.errorbar(bead_diameters['bead_5'][0] / 2, terminal_velocities[4], xerr = bead_diameters['bead_5'][1] / 2, yerr = velocity_errors[4], fmt = 'o', ms = 3.0, label = f'Bead 5')

# Theoretical prediction
rho = 1.26 # g/mm^3
eta = 9.34 # gmm^-1s^-1
plt.plot([bead_diameters[f'bead_{i}'][0] / 2 for i in range(1, 6)], rho * np.array([(bead_diameters[f'bead_{i}'][0] / 2) * terminal_velocities[j - 1] for i in range(1, 6)], dtype = float) / eta, linestyle = '--', color = 'black', label = 'Theoretical Prediction')

# Labels
plt.title('Terminal Velocity vs Bead Radius', fontsize = 12)
plt.xlabel('Bead Radius (mm)', fontsize = 12)
plt.ylabel('Terminal Velocity (mm/s)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()
