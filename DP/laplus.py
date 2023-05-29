import numpy as np
import pandas as pd

def add_laplace_noise(data, sensitivity, epsilon):

    # Calculate the scale parameter for the Laplace distribution
    scale = sensitivity / epsilon

    # Generate noise from the Laplace distribution
    noise = np.random.laplace(loc=0, scale=scale, size=data.shape)

    # Add the noise to the data
    perturbed_data = data + noise

    return perturbed_data



df = pd.read_csv("../test4.csv")
#id=["25","87","93","55","100","73","44","96","116","14","70","35","28","102","33","83","78","95","13","39"]
for i in range(1,128):
    df[str(i)] = add_laplace_noise(np.array(df[str(i)]), 0.1, 0.5)

df.to_csv('../test4-2.csv')


