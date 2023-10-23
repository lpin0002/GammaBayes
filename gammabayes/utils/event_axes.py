
from utils.utils import bkgfull
import numpy as np

bkgfull2d = bkgfull.to_2d()
bkgfull2doffsetaxis = bkgfull2d.axes['offset'].center.value
offsetaxisresolution = bkgfull2doffsetaxis[1]-bkgfull2doffsetaxis[0] # Comes out to 0.2
latbound            = 3.
lonbound            = 3.5



latitudeaxis            = np.linspace(-latbound, latbound, int(round(2*latbound/0.2)))
latitudeaxistrue        = np.linspace(-latbound, latbound, int(round(2*latbound/0.1)))

longitudeaxis           = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.2))) 
longitudeaxistrue       = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.1))) 


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10estart             = np.log10(0.02)
log10eend               = np.log10(300)
log10erange             = log10eend - log10estart
log10eaxis              = np.linspace(log10estart,log10eend,int(np.round(log10erange*50))+1)
log10eaxistrue          = np.linspace(log10estart,log10eend,int(np.round(log10erange*300))+1)



def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob

logjacob = makelogjacob(log10eaxis)
logjacobtrue = makelogjacob(log10eaxistrue)