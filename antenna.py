import matplotlib.pyplot as plt
import numpy as np


def antenna_gain(theta):
	theta_db = 7*np.pi/18
	return -min(12*np.power(theta/theta_db, np.float(2)), 20)
	

angle = np.arange(-np.pi, np.pi, 0.1)
r = [antenna_gain(alpha) for alpha in angle]

# given a dx element, find out the serving eNB
# After finding out the serving eNB, find out its angle 

plt.plot(angle, r, 'ro')
plt.show()
