# plot the static trajectory of the position in time of the bodies
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non interattivo
import matplotlib.pyplot as plt

# Load the data from the file
data = np.loadtxt("../build/positions.txt", delimiter=",")

# Create a new figure
plt.figure()

# Plot the data
plt.plot(data[:, 0], data[:, 1], linewidth=1, label='sun')
plt.plot(data[:, 2], data[:, 3], linewidth=1, label='mercury')
plt.plot(data[:, 4], data[:, 5], linewidth=1, label='venus')
plt.plot(data[:, 6], data[:, 7], linewidth=1, label='earth')
plt.plot(data[:, 8], data[:, 9], linewidth=1, label='mars')
plt.plot(data[:, 10], data[:, 11], linewidth=1, label='jupiter')
plt.plot(data[:, 12], data[:, 13], linewidth=1, label='uranus')
plt.plot(data[:, 14], data[:, 15], linewidth=1, label='neptun')



# Add a legend
plt.legend()
plt.xlim(-3e12, 3e12)  # limite asse x
plt.ylim(-3e12, 3e12)  # limite asse y


# Show the plot      plt.show()
plt.savefig("../build/trajectory.png")  # salva l’immagine

print("Plot salvato in ../build/trajectory.png")

