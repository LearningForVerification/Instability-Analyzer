import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


def generate_distribution_graph(samples, alpha, xi, omega):
    # Plot della distribuzione
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

    # Overlay della PDF teorica
    x = np.linspace(min(samples), max(samples), 1000)
    pdf = skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)
    plt.plot(x, pdf, 'k', linewidth=2)
    plt.title('Distribuzione Normale Asimmetrica')
    plt.xlabel('Valore')
    plt.ylabel('Densità')
    plt.show()


def calculate_nn_dim(params_list):
    d = 784
    n = 4000
    K = 10
    nn_dims = [int(np.ceil((x - K) / (d + 1 + K))) for x in params_list]
    return nn_dims


def get_dims():
    K = 40000
    # Linspace in [15000, 45000]
    values = np.linspace(15000, 45000, num=10)
    samples_mid = values.tolist()

    # Linspace in [3000, 15000]
    values = np.linspace(3000, 15000, num=10)
    samples_low = values.tolist()

    # Linspace in [50000, 750000]
    values = np.linspace(45000, 800000, num=10)
    samples_high = values.tolist()

    params = samples_low + samples_mid + samples_high
    params = [int(x) for x in params]
    nn_dims = calculate_nn_dim(params)
    return nn_dims, K
