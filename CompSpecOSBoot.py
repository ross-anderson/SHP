"""
Code for generating composite spectra from VANDELS survey fits data.
Author: Ross Anderson s1524267
Course: Senior Honours Project 
"""

from astropy.io import fits
#from astropy.stats import bootstrap, sigma_clip
from scipy import optimize
import astropy.stats as s
import numpy as np
import matplotlib.pyplot as plt
import os
import math as m
import time

def attenuation_salim_2018(wl, av, B, delta):
    """
    Returns A(lambda) for the Salim + 2018 dust attenuation
    prescription
    """

    x = 1.0 / wl
    rv_calz = 4.05
    k_calz = (2.659 * (-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3)) + rv_calz

    wl0 = 0.2175
    dwl = 0.035
    d_lam = (B * np.power(wl*dwl, 2)) / (np.power(wl**2-wl0**2, 2) + np.power(wl0*dwl, 2))

    rv_mod = rv_calz / ((rv_calz + 1)*(0.44/0.55)**delta - rv_calz)
    
    kmod = k_calz * (rv_mod/rv_calz) * (wl/0.55)**delta + d_lam

    return (kmod * av) / rv_mod

def chi_fn(x,*params):
    """
    Method to calculate the chi^2 value for a set of data 
    given a set of initial parameters.
    """

    #set variables and parameters
    Av, delta, B = x
    obs, model_in, err, wl = params

    #dust attenuation
    Alambda = attenuation_salim_2018(wl/10000., Av, B, delta)
    model = model_in*(10**(-0.4*Alambda))

    #calculate chi^2 components
    beta_num = np.sum(model*obs/err**2.)
    beta_denom = np.sum((model/err)**2.)
    beta = beta_num/beta_denom
    raw_chi=((beta*model-obs)/err)**2
    chi_sq=np.sum(raw_chi)


    return round(chi_sq,3)

def m_norm(x, *params):
    """
    Method to calculate normalization constant for model flux values
    """
    Av, delta, B = x
    obs, model_in, err, wl = params

    #dust attenuation
    Alambda = attenuation_salim_2018(wl/10000., Av, B, delta)
    model = model_in*(10**(-0.4*Alambda))

    #calculate beta components
    beta_num = np.sum(model*obs/err**2.)
    beta_denom = np.sum((model/err)**2.)
    beta = beta_num/beta_denom
    
    return beta

def main():

    start = time.time()

    # list fits and model files:
    files = os.listdir('./vandels_spectra')
    mfiles = os.listdir('./models2')

    # get mass and redshift information:
    binarr = np.loadtxt('vandels_stellar_masses.dat', dtype=str)
    
    # read in model spectra:
    modl_spec = np.stack([np.loadtxt('./models2/%s' %f, dtype=str) for f in mfiles], axis=0)
    modl_wl = np.asarray(modl_spec[:,:,0], dtype=float)
    modl_flux = np.asarray(modl_spec[:,:,1], dtype=float)
    zlabels = [os.path.splitext(file)[0].split('-')[2] for file in mfiles]
    
    # mask out clean wavelength regions:
    xxwl = np.loadtxt('DirtyLambda2.csv', dtype=float, usecols=0)
    xxwln = np.size(xxwl)
    xxmask = np.isin(modl_wl, xxwl, invert=True)
    cmodl_flux = np.stack([modl_flux[ii][xxmask[ii]] for ii in range(5)])
    cmodl_wl = np.stack([modl_wl[ii][xxmask[ii]] for ii in range(5)])

    # convert model flux from logflam to flam:
    cmodl_flux = (10.**cmodl_flux)
    modl_flux = (10.**modl_flux)
    
    

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

        # mask out dirty wavelengths:
        obs_flux = np.asarray(med_flux)
        maskxx = np.isin(cwl, xxwl, invert=True)
        cobs_flux = obs_flux[maskxx]
        ccwl = cwl[maskxx]
        err = np.asarray(err)
        cerr = err[maskxx]



    

        # execute chi minimisation function:
        mn_chi = []
        adb = []
        ranges=(slice(0., 5., 0.1),slice(-1., 1., 0.05), slice(0., 3., 0.1))
        for i in range(2):
            # extract model values comparable to observed values
            maskobs = np.isin(cmodl_wl[i], ccwl)
            obs_modl_flux = cmodl_flux[i][maskobs]
            data = (cobs_flux, obs_modl_flux, cerr, ccwl)
            chi_data = optimize.brute(chi_fn, ranges, args=data, full_output=True, finish=None)
            mn_chi.append(chi_data[1])
            adb.append(chi_data[0])

        N = mn_chi.index(min(mn_chi))

        # align wavelength grids:
        maskobs = np.isin(modl_wl[N], cwl)
        aligned_modl_flux = modl_flux[N][maskobs]
        aligned_modl_wl = modl_wl[N][maskobs]
        A = adb[N][0]
        D = adb[N][1]
        B = adb[N][2]
        Alambda = attenuation_salim_2018(aligned_modl_wl/10000., A, B, D)
        adb = (A, D, B)
        nmodl_flux = m_norm(adb, obs_flux, aligned_modl_flux, err, aligned_modl_wl)*aligned_modl_flux*(10**(-0.4*Alambda))
        chisq = min(mn_chi)
        rchisq = chisq/(np.size(ccwl)-4.)

        mx = np.amax(np.array([np.amax(nmodl_flux), np.amax(obs_flux)]))
        mn = np.amin(err)
        print(A,D,B)
        
        
        # plot spectrum: 
        hfont = {'fontname':'Liberation Serif'}
        plt.figure(figsize=(20.,10.), edgecolor='black')
        #ax = plt.figure().add_subplot(111)
        plt.plot(cwl, obs_flux, color='black', lw=1.) 
        plt.plot(cwl, nmodl_flux, color='blue', lw=1.)
        plt.plot(cwl, err, color='red', lw=1.)
        plt.ylim(mn-0.25*mx, mx+0.25*mx)
        for i in range(len(ccwl)-1):
            xmin=ccwl[i]-0.5
            xmax=ccwl[i]+0.5
            plt.axvspan(xmin, xmax, alpha=0.25, color='cadetblue', lw=0.)
        plt.xlim(1200, 2000)
        plt.title(r'%s $\leq$ m $\leq$ %s // %s // $\chi^2_\nu$ %s' %(round(lwrm,3),round(uprm,3), zlabels[N], round(rchisq,3)), fontsize=35, **hfont)
        plt.xlabel(r'Rest Wavelength / $\AA$', fontsize=35, **hfont)
        plt.ylabel(r'F$_{\lambda}$ / erg/s/cm$^2$/$\AA$', fontsize=35, **hfont)
        plt.tick_params(axis='both', direction='in', left=True, right=True, top=True, bottom=True, which='both', labelsize=20)
        #plt.show()
        plt.savefig(fname='Vandels_CompSpec2_ModelvsData%s' %j, fmt='png')


    runtime = round(time.time()-start,2)
    print('Runtime: ', runtime, 's')

main()

