from nn import NoiseSchedulerFactory
import matplotlib.pyplot as plt
import numpy as np

def checkScheduler(ns, epochs, noise_std):
    noises = []
    for i in range(epochs):
        noises.append(ns.getNoise())
        ns.step()
    noises = np.array(noises)
    alphas = noises/noise_std
    betas = 1/alphas
    #betas = noises/noise_std
    if not np.isclose(betas.sum(), epochs):
        print(betas.sum())
        print(betas)
        raise Exception()

noise_std = 2
factor = 0.05
epochs = 100
step_start = 2
ns = NoiseSchedulerFactory(noise_std, factor, epochs, step_start, ns_type='step')
noises = []
checkScheduler(ns, epochs, noise_std)

noise_std = 2
factor = 0.6
epochs = 10
step_start = 2
ns = NoiseSchedulerFactory(noise_std, factor, epochs, step_start, ns_type='inc')
checkScheduler(ns, epochs, noise_std)


noise_std = 2
factor = 0.05
epochs = 100
step_start = 96
ns = NoiseSchedulerFactory(noise_std, factor, epochs, step_start, ns_type='revstep')
checkScheduler(ns, epochs, noise_std)

