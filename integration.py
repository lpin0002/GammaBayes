import numpy as np
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from scipy import linalg, stats, integrate, special, interpolate
import chime
from utils2 import axis, makedist, eaxis_mod, logjacob
from dynestyrun import rundynesty


print('\n')
measured = -1
axis = np.linspace(-2,2,axis.shape[0])
logjacob = np.log(10**axis)+np.log(axis[1]-axis[0])+np.log(np.log(10))

logirf = lambda logerecon, logetrue: stats.norm(loc=10**logetrue,scale=2).logpdf(10**logerecon)


# priorfull = lambda logetrue: prior(logetrue)-np.log(integrate.trapezoid(y = np.exp(prior(axis)), x=10**axis))
priorfull = makedist(1.1*measured)




res = rundynesty(measured, priorfull, logirf, axis=axis)
chime.info('sonic')

# dE = E ln(10) d(log_10(E))
postvals = np.array([logirf(measured, logeval)-special.logsumexp(logirf(axis, logeval)+logjacob) for logeval in axis])+priorfull(axis)+logjacob
plt.figure()
histvals = plt.hist(res.samples, bins=axis)
plt.plot(axis, np.exp(postvals)/np.max(np.exp(postvals))*np.max(histvals[0]), c='ForestGreen')
plt.axvline(measured, lw=0.5, c='r')
plt.show()

lnz_direct = special.logsumexp(postvals)
print(f'\nlog direct = {lnz_direct},log nested = {res.logz[-1]}, ratio=nested/direct={np.exp(res.logz[-1]-lnz_direct)}')

fig = plt.subplots(4,1, figsize=(14,7))
fig, axes = dyplot.runplot(res, lnz_truth=lnz_direct, fig=fig)
plt.savefig("ConvergencePlot.pdf")
plt.show()
print('\n')

chime.success()