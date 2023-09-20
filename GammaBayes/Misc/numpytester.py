import numpy as np


# Generate some example data
nbins = 5
nsamples = 3
lambdarange = np.random.rand(nbins, 1)
lambdarange = np.matrix(np.linspace(0,1,nbins))
sigmarglogzvals = np.matrix(np.linspace(0,10,nsamples))

# Add each element of lambdarange to sigmarglogzvals
print(lambdarange.shape)
print(sigmarglogzvals.shape)

result = lambdarange + sigmarglogzvals.T

print(result)