# 
from gammabayes.BFCalc.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput, energymassinputspectralfunc
from gammabayes.utils import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis, angularseparation, convertlonlat_to_offset
from gammabayes.utils import psf_efficient, edisp_efficient, tqdm, logjacob, resources_dir
from gammabayes.utils import irfs

from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy import units as u
from scipy import special,stats
from scipy.integrate import simps
from matplotlib import cm
from tqdm.autonotebook import tqdm as notebook_tqdm
import os, sys
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing

from gammapy.datasets import Datasets, MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)


aeff = irfs['aeff']

aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value


try:
    run_function = int(sys.argv[1])
except:
    run_function = 0
    
try:
    setup_irfnormalisations = int(sys.argv[2])
except:
    setup_irfnormalisations = 1
    
try:
    setup_astrobkg = int(sys.argv[3])
except:
    setup_astrobkg = 1

def setup(setup_irfnormalisations=1, setup_astrobkg=1, log10eaxistrue=log10eaxistrue, log10eaxis=log10eaxis, 
          longitudeaxistrue=longitudeaxistrue, longitudeaxis=longitudeaxis, latitudeaxistrue=latitudeaxistrue, latitudeaxis=latitudeaxis,
          logjacob=logjacob, save_directory = resources_dir, psf=psf_efficient, edisp=edisp_efficient, aeff=aefffunc):
    def powerlaw(energy, index, phi0=1):
        return phi0*energy**(index)


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    if setup_irfnormalisations:
        psfnorm = []
        for logeval in tqdm(log10eaxistrue, desc='Setting up psf normalisation', ncols=80):
            psflogerow = []
            for lonval in longitudeaxistrue:
                log10eaxistrue_mesh, longitudeaxistrue_mesh, latitudeaxistrue_mesh, longitudeaxis_mesh, latitudeaxis_mesh  = np.meshgrid(logeval,
                                                                                                                                        lonval, 
                                                                                                                                        latitudeaxistrue, 
                                                                                                                                        longitudeaxis, 
                                                                                                                                        latitudeaxis, indexing='ij')

                truecoords = np.array([longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()])

                recon_coords = np.array([longitudeaxis_mesh.flatten(), latitudeaxis_mesh.flatten()])

                rad = angularseparation(recon_coords, truecoords)
                offset = convertlonlat_to_offset(truecoords)

                psfvals = psf_efficient(rad, log10eaxistrue_mesh.flatten(), offset).reshape(log10eaxistrue_mesh.shape)
                
                psfnormvals = special.logsumexp(psfvals, axis=(-2,-1))
                
                psflogerow.append(psfnormvals)
            psfnorm.append(psflogerow)
                

    # 
        psfnorm = np.squeeze(np.array(psfnorm))



        edispnorm = []
        for logeval in tqdm(log10eaxistrue, desc='Setting up edisp normalisation', ncols=80):
            log10eaxistrue_mesh, longitudeaxistrue_mesh, latitudeaxistrue_mesh, log10eaxis_mesh  = np.meshgrid(logeval,
                                                                                                                longitudeaxistrue, 
                                                                                                                latitudeaxistrue, 
                                                                                                                log10eaxis,
                                                                                                                indexing='ij')

            truecoords = np.array([longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()])
            
            offset = convertlonlat_to_offset(truecoords)

            edispvals = np.squeeze(edisp_efficient(log10eaxis_mesh.flatten(), log10eaxistrue_mesh.flatten(), offset).reshape(log10eaxistrue_mesh.shape))
                
            edispnormvals = np.squeeze(special.logsumexp(edispvals+logjacob, axis=-1))
            
            edispnorm.append(edispnormvals)


        edispnorm = np.array(edispnorm)

        edispnorm[np.isneginf(edispnorm)] = 0
        psfnorm[np.isneginf(psfnorm)] = 0


        np.save(save_directory+"/psfnormalisation.npy", psfnorm)
        np.save(save_directory+"/edispnormalisation.npy", edispnorm)

    if setup_astrobkg:
        print("Setting up the astrophysical background\n\n")

        print("Setting up HESS sources\n")
        ### HESS Source Background

        from gammapy.catalog import SourceCatalogHGPS

        print("Accessing HESS Catalogue")
        hess_catalog = SourceCatalogHGPS(resources_dir+"/hgps_catalog_v1.fits.gz")

        hess_models = hess_catalog.to_models()

        print(f"\nThere are {len(hess_models)} sources within the HGPS.")

        print("Constructing geometry which the models will be applied to")
        trueenergyaxis = 10**log10eaxistrue*u.TeV
        print(log10eaxistrue.shape)


        energy_axis_true = MapAxis.from_nodes(trueenergyaxis, interp='log', name="energy_true")
        print("Background energy axis check shape: \n", energy_axis_true, '\nvs\n', trueenergyaxis.shape)
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
            filename=resources_dir+"/gll_iem_v06_gc.fits.gz", normalize=False
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
        

        lonmesh, energymesh, latmesh  = np.meshgrid(fermilonaxis, fermienergyvals, fermilataxis)

        aefftable = aeff(energymesh, np.sqrt((lonmesh**2)+(latmesh**2)))

        combinedplotmapwithaeff = aefftable*combinedplotmap
        combinedplotmapwithaeff = combinedplotmapwithaeff

        topbound=1e500
        combinedplotmapwithaeff[combinedplotmapwithaeff>topbound] = topbound
        # combinedplotmapwithaeff=combinedplotmapwithaeff/normalisation
        # modtopbound = topbound/normalisation
        spatialplotcombined = np.sum((combinedplotmapwithaeff.T*10**log10eaxistrue).T, axis=0)


        print("\n\nSaving the final result")
        np.save(save_directory+"/unnormalised_astrophysicalbackground.npy", combinedplotmapwithaeff)
            



if run_function:
    setup(setup_astrobkg=setup_astrobkg, setup_irfnormalisations=setup_irfnormalisations)