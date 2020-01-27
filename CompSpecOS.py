"""
Code for generating composite spectra from VANDELS survey fits data.

Author: Ross Anderson s1524267
Course: Senior Honours Project 

"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def main():

    start = time.time()

    # list fits files:
    files = os.listdir('./vandels_spectra')
    
    # extract flux and flux errors (units are ergs/s/cm2/A):
    fluxes = np.stack([fits.open('./vandels_spectra/%s' %f)['EXR1D'].data for f in files], axis=0)
    flux_errors = np.stack([fits.open('./vandels_spectra/%s' %f)['NOISE'].data for f in files], axis=0)

    # extract wavelength information:
    wl0s = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['CRVAL1'] for f in files], axis=0) # initial wavelength
    dwls = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['CDELT1'] for f in files], axis=0) # wavelength interval
    naxis = fluxes.shape[1] # number of data points
    wlfs = wl0s + (dwls * naxis) # final wavelength
    
    # get redshifts and flags:
    redshifts_raw = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['HIERARCH PND Z'] for f in files], axis=0) 
    redshifts = redshifts_raw.astype(float)
    flags_raw = np.stack([fits.open('./vandels_spectra/%s' %f)[0].header['HIERARCH PND ZFLAGS'] for f in files], axis=0)
    flags = flags_raw.astype(float)
    
    # create rest frame wavelength grids( GENERALISE THIS LATER ):
    wl0 = np.mean(wl0s)
    dwl = np.mean(dwls)
    wlf = np.mean(wlfs)
    
    wl_obs = np.arange(wl0, wlf, dwl)
    wl_obsarr = np.tile(wl_obs, (200,1))
    wl_rest = np.stack([wl_obsarr[i,:] /(1. + redshifts[i]) for i in range(np.size(redshifts))], axis=0) # shift all to rest frame
    
    
    flux_stack = np.median(fluxes, axis=0)
    wl_stack = np.median(wl_rest, axis=0)
    wl_stack = np.delete(wl_stack, 2154)

    print(np.amax(redshifts), np.amin(redshifts))

    # plot spectrum:
    plt.plot(wl_stack, flux_stack, color='black', lw=1.) 
    #plt.plot(wl_rest, flux_error, color='red', lw=1.)
    plt.ylim(np.amin(flux_stack), np.amax(flux_stack))
    plt.xlabel('Wavelength / A', fontsize=15)
    plt.ylabel('Flux / erg/s/cm^2/A', fontsize=15)
    plt.show()


    runtime = round(time.time()-start,2)
    print('Runtime: ', runtime, 's')

main()



