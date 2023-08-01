from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from utils3d import longitudeaxistrue, latitudeaxistrue, log10eaxistrue, logjacob, longitudeaxis, latitudeaxis, log10eaxis, edisp, psf
from scipy import special

# %matplotlib inline
from gammapy.datasets import Datasets, MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)


# %matplotlib inline
import numpy as np
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom

################################################################################################################
################################################################################################################
#### Setting up the IRF matrices



print("\n\n\nSetting up the meshgrid for the psf matrix construction\n")
lontrue_mesh_psf, logetrue_mesh_psf, lattrue_mesh_psf, lonrecon_mesh_psf, latrecon_mesh_psf = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue, longitudeaxis, latitudeaxis)

print("Constructing the point spread function matrix")
psfmatrix = psf(np.array([lonrecon_mesh_psf.flatten(), latrecon_mesh_psf.flatten()]), np.array([lontrue_mesh_psf.flatten(), lattrue_mesh_psf.flatten()]), logetrue_mesh_psf.flatten()).reshape(logetrue_mesh_psf.shape)

print("\n\nSetting up the meshgrid for the edisp matrix construction\n")
lontrue_mesh_edisp, logetrue_mesh_edisp, lattrue_mesh_edisp, logerecon_mesh_edisp,  = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue, log10eaxis)

print("Constructing the energy dispersion matrix")
edispmatrix = edisp(logerecon_mesh_edisp.flatten(), logetrue_mesh_edisp.flatten(), np.array([lontrue_mesh_edisp.flatten(), lattrue_mesh_edisp.flatten()])).reshape(logetrue_mesh_edisp.shape)

print("\n\nNormalising the matrices")
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


print("\n\nSaving the IRF Matrices\n\n\n")
np.save("psfmatrix.npy", psfmatrix)
np.save("edispmatrix.npy", edispmatrix)




############################################################################################################################################
############################################################################################################################################
### Setting up the astrophysical background
print("Setting up the astrophysical background\n\n")

print("Setting up HESS sources\n")
### HESS Source Background

from gammapy.catalog import SourceCatalogHGPS

print("Accessing HESS Catalogue")
hess_catalog = SourceCatalogHGPS("hgps_catalog_v1.fits.gz")

hess_models = hess_catalog.to_models()

print(f"\nThere are {len(hess_models)} sources within the HGPS.")

print("Constructing geometry which the models will be applied to")
trueenergyaxis = 10**log10eaxistrue*u.TeV

energy_axis_true = MapAxis.from_nodes(trueenergyaxis, interp='log', name="energy_true")

goodgeom = WcsGeom.create(
    skydir=SkyCoord(0, 0, unit="deg", frame='galactic'),
    binsz=0.2,
    width=(longitudeaxistrue[-1]-longitudeaxistrue[0]+0.2, latitudeaxistrue[-1]-latitudeaxistrue[0]+0.2),
    frame="galactic",
    proj="CAR",
    axes=[energy_axis_true],
)

goodmap = Map.from_geom(goodgeom)

hessenergyaxis = energy_axis_true.center.value




# fig, ax = plt.subplots(2,4, figsize=(8,3))


print("Constructing the contribution to the background prior from the HESS Catalogue")
m= goodmap
count=0
fullhessdataset = 0
for idx, model in enumerate(hess_models):
    templvalue = model.spatial_model.position.l.value
    tempbvalue = model.spatial_model.position.b.value
    
    if templvalue>180:
        templvalue=templvalue-360
    
    if np.abs(templvalue)<5 and np.abs(tempbvalue)<5:
        print(templvalue, tempbvalue, idx)
        try:
            m.quantity = model.evaluate_geom(m.geom)
            # m.plot(ax=ax[count//4, count%4], add_cbar=True)
            startdata = m.data
            data = startdata[~np.isnan(startdata)]
            if data.size!=0:
                data = data.reshape(startdata.shape)
                fullhessdataset+=data
            
        except:
            print("Something weird happened")
            print(idx, '\n\n')
        count+=1
# plt.show()
fullhessdataset = np.transpose(fullhessdataset, axes=(0,2,1))


hessgeom = goodmap.geom
hesslonvals = hessgeom.get_coord().lon.value[0][0]
hesslonvals = hesslonvals[::-1]
hesslonvals[hesslonvals>180] = hesslonvals[hesslonvals>180]-360

hesslatvals = hessgeom.get_coord().lat.value[0][:,0]




print("\n\nSetting up the Fermi-LAT diffuse background\n")
### Fermi-LAT Diffuse Background

template_diffuse = TemplateSpatialModel.read(
    filename="gll_iem_v06_gc.fits.gz", normalize=False
)

print(template_diffuse.map)

diffuse_iem = SkyModel(
    spectral_model=PowerLawNormSpectralModel(),
    spatial_model=template_diffuse,
    name="diffuse-iem",
)


print("Constructing the Fermi-LAT background")
fermievaluated = np.transpose(diffuse_iem.evaluate_geom(goodgeom), axes=(0,2,1))
fermievaluated = fermievaluated.to(1/u.TeV/u.s/u.sr/(u.cm)**2)


fermilonaxistemp = np.unique(goodgeom.to_image().get_coord().lon.value)
firstover180idx = np.where(fermilonaxistemp>180)[0][0]
fermilonaxistemp[fermilonaxistemp>180] = fermilonaxistemp[fermilonaxistemp>180]-360
fermilonaxistemp.sort()
fermilonaxis = fermilonaxistemp


fermilataxis = goodgeom.get_coord().lat.value[0][:,0]
fermiunit = fermievaluated.unit
fermienergyvals = energy_axis_true.center.value
fermiplotmap = fermievaluated.value



print("\n\nCombining the HESS and Fermi-LAT backgrounds")
combinedplotmap = np.flip(fermiplotmap + fullhessdataset, axis=1)


print("\n\nApplying the effective area to the maps")
from utils3d import irfs
aeff = irfs['aeff']

aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value

lonmesh, energymesh, latmesh  = np.meshgrid(fermilonaxis, fermienergyvals, fermilataxis)

aefftable = aefffunc(energymesh, np.sqrt((lonmesh**2)+(latmesh**2)))

combinedplotmapwithaeff = aefftable*combinedplotmap
combinedplotmapwithaeff = combinedplotmapwithaeff
topbound=1e100
combinedplotmapwithaeff[combinedplotmapwithaeff>topbound] = topbound
normalisation = np.sum(combinedplotmapwithaeff.T*10**log10eaxistrue)
# combinedplotmapwithaeff=combinedplotmapwithaeff/normalisation
# modtopbound = topbound/normalisation
spatialplotcombined = np.sum((combinedplotmapwithaeff.T*10**log10eaxistrue).T, axis=0)
spatialplotcombinedmaxvalue = np.max(spatialplotcombined)


print("\n\nSaving the final result")
np.save("unnormalised_astrophysicalbackground.npy", combinedplotmapwithaeff)