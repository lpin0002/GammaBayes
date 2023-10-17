import numpy as np
from os import path
from gammabayes.utils.utils import makelogjacob, bkgfull, psf3d, create_axes
resources_dir = path.join(path.dirname(__file__), '../package_data')




bkgfull2d = bkgfull.to_2d()
bkgfull2doffsetaxis = bkgfull2d.axes['offset'].center.value
offsetaxis = psf3d.axes['rad'].center.value
offsetaxisresolution = bkgfull2doffsetaxis[1]-bkgfull2doffsetaxis[0] # Comes out to 0.2
latbound            = 3.
lonbound            = 3.5
log10estart             = -1.0
log10eend               = 2.0


log10eaxistrue,longitudeaxistrue,latitudeaxistrue = create_axes(log10estart, log10eend, 
                     200, 0.2, 
                     -lonbound, +lonbound,
                     -latbound, +latbound)
log10eaxis,longitudeaxis,latitudeaxis= create_axes(log10estart, log10eend, 
                     50, 0.4, 
                     -lonbound, +lonbound,
                     -latbound, +latbound)


logjacob = makelogjacob(log10eaxis)
logjacobtrue = makelogjacob(log10eaxistrue)

astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysicalbackground.npy")
psfnormalisationvalues = np.load(resources_dir+"/psfnormalisation.npy")
edispnormalisationvalues = np.load(resources_dir+"/edispnormalisation.npy")
    