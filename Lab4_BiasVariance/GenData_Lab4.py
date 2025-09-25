import numpy as np
import random

# Define model parameters:
x_min = 0
x_max = 6
x_interval = 0.02
slope = 1.0
freq = 3.0

npts = 50
y_obs_sigma = 0.25

# f(x) = slope*x + sin(freq*x)
def f(x):
    y = slope*x + np.sin(freq*x)
    return y

# y_obs = f(x) + N(0, sigma^2)    
def genNoisyData():
    x_obs = np.linspace(x_min, x_max, npts)
    np.random.shuffle(x_obs)
    y_obs = f(x_obs) + y_obs_sigma*np.random.randn(len(x_obs))
    return (x_obs, y_obs)
