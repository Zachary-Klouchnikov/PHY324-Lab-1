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
MAIN
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

"Plotting Time vs. Position for Glycerine Data"
fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (6, 10))

# Plotting data for bead 1
for key in glycerine_path[0 : 5]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[0].scatter(time, position, linewidths = 0, label = f'{key}')

# Labels
axes[0].set_title('Bead 1 Trials', fontsize = 12)

axes[0].legend(fontsize = 12)
axes[0].grid()

# Plotting data for bead 2
for key in glycerine_path[5 : 10]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[1].scatter(time, position, linewidths = 0, label = f'{key}')

# Labels
axes[1].set_title('Bead 2 Trials', fontsize = 12)

axes[1].legend(fontsize = 12)
axes[1].grid()

# Plotting data for bead 3
for key in glycerine_path[10 : 15]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[2].scatter(time, position, linewidths = 0, label = f'{key}')

# Labels
axes[2].set_title('Bead 3 Trials', fontsize = 12)
axes[2].set_ylabel('Position (mm)', fontsize = 12)

axes[2].legend(fontsize = 12)
axes[2].grid()

# Plotting data for bead 4
for key in glycerine_path[15 : 20]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[3].scatter(time, position, linewidths = 0, label = f'{key}')

# Labels
axes[3].set_title('Bead 4 Trials', fontsize = 12)

axes[3].legend(fontsize = 12)
axes[3].grid()

# Plotting data for bead 5
for key in glycerine_path[20 : 25]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    axes[4].scatter(time, position, linewidths = 0, label = f'{key}')

# Labels
axes[4].set_title('Bead 5 Trials', fontsize = 12)
axes[4].set_xlabel('Time (s)', fontsize = 12)

axes[4].legend(fontsize = 12)
axes[4].grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.tight_layout()
plt.show()

"Plotting the Analysis of Glycerin Data for Each Bead"
# Trials for Bead 1
plt.figure()

# Analysing and plotting data for each trial
slopes = []
intercepts = []
for key in glycerine_path[0 : 5]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 30s and 50s
    filtered_indices = np.where((time >= 30) & (time <= 50))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    intercepts.append(popt[1])

    plt.scatter(filtered_time, filtered_position, linewidths = 0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}x + {np.mean(intercepts):.2f}')

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
intercepts = []
for key in glycerine_path[5 : 10]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 15s and 50s
    filtered_indices = np.where((time >= 15) & (time <= 50))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    intercepts.append(popt[1])

    plt.scatter(filtered_time, filtered_position, linewidths = 0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}x + {np.mean(intercepts):.2f}')

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
intercepts = []
for key in glycerine_path[10 : 15]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 10s and 40s
    filtered_indices = np.where((time >= 10) & (time <= 40))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    intercepts.append(popt[1])

    plt.scatter(filtered_time, filtered_position, linewidths = 0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}x + {np.mean(intercepts):.2f}')

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
intercepts = []
for key in glycerine_path[15 : 20]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 10s and 40s
    filtered_indices = np.where((time >= 5) & (time <= 17.5))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    intercepts.append(popt[1])

    plt.scatter(filtered_time, filtered_position, linewidths = 0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}x + {np.mean(intercepts):.2f}')

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
intercepts = []
for key in glycerine_path[20 : 25]:
    time = glycerine_data[key][0]
    position = glycerine_data[key][1]

    # Filtering data for times between 10s and 40s
    filtered_indices = np.where((time >= 3) & (time <= 11))
    filtered_time = time[filtered_indices]
    filtered_position = position[filtered_indices]

    # Fitting data to linear function
    popt, pcov = curve_fit(linear_fit, filtered_time, filtered_position)
    slopes.append(popt[0])
    intercepts.append(popt[1])

    plt.scatter(filtered_time, filtered_position, linewidths = 0, label = f'{key}')
    plt.plot(filtered_time, linear_fit(filtered_time, popt[0], popt[1]), linestyle = '--')

# Average fit line
plt.plot(filtered_time, linear_fit(filtered_time, np.mean(slopes), np.mean(intercepts)), color = 'black', linestyle = '-', label = f'Average Fit: y = {np.mean(slopes):.2f}x + {np.mean(intercepts):.2f}')

# Labels
plt.title('Bead 5 Trials', fontsize = 12)
plt.xlabel('Time (s)', fontsize = 12)
plt.ylabel('Position (mm)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()
