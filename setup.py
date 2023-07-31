from utils3d import psf, edisp, longitudeaxis, longitudeaxistrue, latitudeaxis, latitudeaxistrue, log10eaxis, log10eaxistrue, logjacob
import numpy as np
from scipy import special


# Setting up the meshgrid for the psf matrix construction
lontrue_mesh_psf, logetrue_mesh_psf, lattrue_mesh_psf, lonrecon_mesh_psf, latrecon_mesh_psf = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue, longitudeaxis, latitudeaxis)

# Constructing the point spread function matrix
psfmatrix = psf(np.array([lonrecon_mesh_psf.flatten(), latrecon_mesh_psf.flatten()]), np.array([lontrue_mesh_psf.flatten(), lattrue_mesh_psf.flatten()]), logetrue_mesh_psf.flatten()).reshape(logetrue_mesh_psf.shape)

# Setting up the meshgrid for the edisp matrix construction
lontrue_mesh_edisp, logetrue_mesh_edisp, lattrue_mesh_edisp, logerecon_mesh_edisp,  = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue, log10eaxis)

# Constructing the energy dispersion matrix
edispmatrix = edisp(logerecon_mesh_edisp.flatten(), logetrue_mesh_edisp.flatten(), np.array([lontrue_mesh_edisp.flatten(), lattrue_mesh_edisp.flatten()])).reshape(logetrue_mesh_edisp.shape)

# Normalising the matrices
psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))
edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)

# Ignoring -np.inf values
edispnormalisation[edispnormalisation==-np.inf] = 0
psfnormalisation[psfnormalisation==-np.inf] = 0   

# Applying the normalisations to the matrices
edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]


# Normalisaing twice to reduce any computation instability
psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))
edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)

# Ignoring -np.inf values
edispnormalisation[edispnormalisation==-np.inf] = 0
psfnormalisation[psfnormalisation==-np.inf] = 0   

# Applying the normalisations to the matrices again
edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]


# Saving the matrices
np.save("psfmatrix.npy", psfmatrix)
np.save("edispmatrix.npy", edispmatrix)


