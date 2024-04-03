import matplotlib.pyplot as plt
import numpy as np

# Range away from subleading jet
eta_range_start = -2.4
eta_range_end   = 2.4
phi_range_start = -2.4
phi_range_end   = 2.4

cmap = plt.get_cmap('gray_r')

nevents = 50000


# Jet image size
MAX_HEIGHT, MAX_WIDTH = 32, 44

cmap = plt.get_cmap('viridis')
nevents = 50000