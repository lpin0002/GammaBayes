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
import sys
def powerlaw(energy, index, phi0=1):
    return phi0*energy**(index)

try:
    irfsetup = int(sys.argv[1])
except:
    irfsetup = 1
try:
    bkgsetup = int(sys.argv[2])
except:
    bkgsetup = 1

################################################################################################################
################################################################################################################
#### Setting up the IRF matrices


if irfsetup:
    print("\n\n\nSetting up the meshgrid for the psf matrix construction\n")
    logetrue_mesh_psf, lontrue_mesh_psf, lattrue_mesh_psf, lonrecon_mesh_psf, latrecon_mesh_psf = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, longitudeaxis, latitudeaxis, indexing='ij')

    print("Constructing the point spread function matrix")
    psfmatrix = psf(np.array([lonrecon_mesh_psf.flatten(), latrecon_mesh_psf.flatten()]), logetrue_mesh_psf.flatten(), np.array([lontrue_mesh_psf.flatten(), lattrue_mesh_psf.flatten()])).reshape(logetrue_mesh_psf.shape)

    print("\n\nSetting up the meshgrid for the edisp matrix construction\n")
    logetrue_mesh_edisp, lontrue_mesh_edisp, lattrue_mesh_edisp, logerecon_mesh_edisp,  = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, indexing='ij')

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

    print("\n\nSaving the IRF Matrices")
    np.save("psfmatrix.npy", psfmatrix)
    np.save("edispmatrix.npy", edispmatrix)

    print(f'\n\n psfmatrix shape: {psfmatrix.shape}, edispmatrix shape: {edispmatrix.shape}\n\n')




############################################################################################################################################
############################################################################################################################################
### Setting up the astrophysical background

if bkgsetup:
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
    skyresolution = longitudeaxistrue[1]-longitudeaxistrue[0]
    lonwidth = longitudeaxistrue[-1]-longitudeaxistrue[0]
    latwidth = latitudeaxistrue[-1]-latitudeaxistrue[0]
    goodgeom = WcsGeom.create(
        skydir=SkyCoord(0, 0, unit="deg", frame='galactic'),
        binsz=skyresolution,
        width=(lonwidth+skyresolution, latwidth+skyresolution), # End bound is non-inclusive but for symmetry we require it to be
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
    fermievaluated = np.flip(np.transpose(diffuse_iem.evaluate_geom(goodgeom), axes=(0,2,1)), axis=1)
    fermiaveraged = special.logsumexp(np.log(fermievaluated.value.T)+10**log10eaxistrue+np.log(np.log(10))+log10eaxistrue[1]-log10eaxistrue[0], axis=2).T

    fermiaveraged = fermiaveraged-special.logsumexp(fermiaveraged+np.log(longitudeaxistrue[1]-longitudeaxistrue[0])+np.log(latitudeaxistrue[1]-latitudeaxistrue[0]))
    fermifull = np.exp(fermiaveraged[np.newaxis, :, :]+np.log(powerlaw(10**log10eaxistrue, index=-2.41, phi0=1.36*1e-8))[:, np.newaxis, np.newaxis])


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
    combinedplotmap = fermifull #+ np.flip(fullhessdataset, axis=1)




    print("\n\nApplying the effective area to the maps")
    from utils3d import irfs
    aeff = irfs['aeff']

    aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value

    lonmesh, energymesh, latmesh  = np.meshgrid(fermilonaxis, fermienergyvals, fermilataxis)

    aefftable = aefffunc(energymesh, np.sqrt((lonmesh**2)+(latmesh**2)))

    combinedplotmapwithaeff = aefftable*combinedplotmap
    combinedplotmapwithaeff = combinedplotmapwithaeff

    topbound=1e500
    combinedplotmapwithaeff[combinedplotmapwithaeff>topbound] = topbound
    normalisation = np.sum(combinedplotmapwithaeff.T*10**log10eaxistrue)
    # combinedplotmapwithaeff=combinedplotmapwithaeff/normalisation
    # modtopbound = topbound/normalisation
    spatialplotcombined = np.sum((combinedplotmapwithaeff.T*10**log10eaxistrue).T, axis=0)
    spatialplotcombinedmaxvalue = np.max(spatialplotcombined)


    print("\n\nSaving the final result")
    np.save("unnormalised_astrophysicalbackground.npy", combinedplotmapwithaeff)
    print(f"\n\nFinal output shape: {combinedplotmapwithaeff.shape}")
    
print("Script Finished!")