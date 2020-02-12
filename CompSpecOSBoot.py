"""
Code for generating composite spectra from VANDELS survey fits data.
Author: Ross Anderson s1524267
Course: Senior Honours Project 
"""

from astropy.io import fits
#from astropy.stats import bootstrap, sigma_clip
import astropy.stats as s
import numpy as np
import matplotlib.pyplot as plt
import os
import math as m
import time

def chi_fn(x,*params):
    """
    Method to calculate the chi^2 value for a set of data 
    given a set of initial parameters.
    """

    #Set variables and parameters
    Av, delta, B = x
    obs, model, err = data


    #calculate chi^2 components
    raw_chi=((model-data)/err)**2
    chi_fn.raw_chi_sq=np.sum(raw_chi, axis=1)


    #calculate min chi^2
    chi_sq=np.amin(chi_fn.raw_chi_sq)


    return round(chi_sq,3)

def main():

    start = time.time()

    # list fits and model files:
    files = os.listdir('./vandels_spectra')
    mfiles = os.listdir('./models')

    # get mass and redshift information:
    binarr = np.loadtxt('vandels_stellar_masses.dat', dtype=str)
    
    # read in model spectra:
    mod_lflux = np.stack([np.loadtxt('./models/%s' %f, dtype=str) for f in mfiles], axis=0)
    zlabels = [os.path.splitext(file)[0] for file in mfiles]

    # mask out clean wavelength regions:
    xxwl = np.loadtxt('DirtyLambda.csv', dtype=float, usecols=0)
    xxwln = np.size(xxwl)
    print(np.size(mod_lflux, axis=0))
    mask = [mod_lflux[:,:,0][ii] != xxwl for ii in range(np.size(mod_lflux, axis=0))]
    
    print(np.shape(mod_lflux[mask]))
    
main()
"""
    # create solar mass bins:
    fileno = np.size(binarr, axis=0)
    binno = 10.
    massarr = np.sort(np.asfarray([binarr[i][2] for i in range(fileno)]))
    mass_range = np.ptp(massarr)
    parse = fileno/binno
    binmin = 0
    binn = {}
    for j in range(int(binno)):
        binn[j] = []
        lwr = round(binmin + j*parse)
        upr = round(binmin + (j+1)*parse)-1
        print(upr-lwr)
        lwrm = massarr[lwr]
        uprm = massarr[upr]
        for i in range(fileno):
            mass = float(binarr[i][2])
            filename = binarr[i][0]
            if lwrm <= mass <= uprm:
                binn[j].append(filename)
    

    
        # extract flux and flux errors (units are ergs/s/cm2/A):
        fluxes = np.stack([fits.open('./vandels_spectra/%s' %f)['EXR1D'].data for f in binn[j]], axis=0)
        flux_errors = np.stack([fits.open('./vandels_spectra/%s' %f)['NOISE'].data for f in binn[j]], axis=0)
    
      
        # extract wavelength information:
        wl0s = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['CRVAL1'] for f in binn[j]], axis=0) # initial wavelength
        dwls = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['CDELT1'] for f in binn[j]], axis=0) # wavelength interval
        naxis = fluxes.shape[1] # number of data points
        wlfs = wl0s + (dwls * naxis) # final wavelength
    
        # get redshifts and flags:
        redshifts_raw = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['HIERARCH PND Z'] for f in binn[j]], axis=0) 
        redshifts = redshifts_raw.astype(float)
        flags_raw = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['HIERARCH PND ZFLAGS'] for f in binn[j]], axis=0)
        flags = flags_raw.astype(float)
    
        # create rest frame wavelength grids:
        wl0 = wl0s[0]
        dwl = dwls[0]
        wlf = wlfs[0]
    
        wl_obs = np.arange(wl0, wlf, dwl)
        wl_obsarr = np.tile(wl_obs, (len(binn[j]),1))
        wl_rest = np.stack([wl_obsarr[i,:] /(1. + redshifts[i]) for i in range(np.size(redshifts))], axis=0) # shift all to rest frame
        wl_rest = wl_rest.flatten()
        fluxes = fluxes.flatten()
    
        # create common wavelength grid:
        cwl = np.arange(1000, 2000, 1)
    

        # find median flux values:
        med_flux = []
        err = []
        for k in range(1000, 2000, 1):
            mask = (k < wl_rest) & (wl_rest < k+1)
            clippedfluxes = s.sigma_clip(fluxes[mask], masked=False)
            median = np.median(clippedfluxes)
            med_flux.append(median)
            bootarr = s.bootstrap(clippedfluxes, 100)
            bootmed = np.median(bootarr, axis=1)
            sigma = np.std(bootmed)
            err.append(sigma)
        
        mx = max(med_flux)
        mn = min(med_flux)


        # execute chi minimisation function:
        ranges=(slice(0., 5., 1.),slice(-1., 1., 0.5.), slice(0., 3., 1.))
        data = (med_flux, , err)
        chi_data = optimize.brute(chi_fn, ranges, args=data, full_output=True, finish=None)
    
        
        # plot spectrum: 
        plt.figure(figsize=(20.,10.), edgecolor='black')
        plt.plot(cwl, med_flux, color='black', lw=1.) 
        plt.plot(cwl, err, color='red', lw=1.)
        plt.ylim(mn-0.25*mx, mx+0.25*mx)
        plt.title(r'Composite Spectrum: %s $\leq$ m $\leq$ %s' %(round(lwrm,3),round(uprm,3)))
        plt.xlabel(r'Wavelength / $\AA$', fontsize=15)
        plt.ylabel(r'Flux / erg/s/cm$^2$/$\AA$', fontsize=15)
        plt.tick_params(axis='both', direction='in', left=True, right=True, top=True, bottom=True, which='both')
        #plt.show()
        plt.savefig(fname='Vandels_CompSpec_%s' %j, fmt='png')


    runtime = round(time.time()-start,2)
    print('Runtime: ', runtime, 's')

main()

"""
