"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
"""
FUNCTIONS
"""

"""
MAIN
"""

time, position = np.loadtxt('data/bead_1_trial_2.txt', dtype = float, delimiter = '\t', skiprows = 2, unpack = True)

"Plotting Total Absolute Frictional Force"
plt.figure()

# Plotting total absolute frictional force
plt.scatter(time, position, color = 'teal', label = 'Position vs Time')

# # Labels
# plt.title("Total Absolute Frictional Force", fontsize = 12)
# plt.xlabel("Velocity of the Mass (m/s)", fontsize = 12)
# plt.ylabel("Force (N)", fontsize = 12)

# plt.legend(fontsize = 12)
# plt.grid()

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()
