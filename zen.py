#! /usr/bin/env python

# Implementation of PLD algorithm
# Team members:
#   Ryan Challener (rchallen@knights.ucf.edu)
#   Em DeLarme
#   Andrew Foster
#
# History:
#   2015-11-02 em       Initial implementation
#   2015-11-20 rchallen Updates for use with POET
#   2016-10-07 rchallen Plotting functions
#   2016-02-20 rchallen Lots of bug fixes. Made git repo.

import os
import sys
import time
import numpy as np
import scipy.optimize as sco
import zen_funcs as zf
import zenplots as zp
import matplotlib.pyplot as plt
import ConfigParser
sys.path.insert(1, "./mccubed/")
sys.path.insert(2, "./poetlib")
import MCcubed as mc3
import manageevent as me

def main():
    '''
    One function to rule them all.
    '''

    # Parse the command line arguments
    eventname = sys.argv[1]
    cfile     = sys.argv[2]

    outdir = time.strftime('%Y-%m-%d-%H:%M:%S') + '_' + eventname 

    try:
        outdirext = sys.argv[3]
        outdir = outdir + '-' + outdirext + '/'
    except:
        outdir = outdir + '/'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logfile = 'zen.log'
    lf = open(outdir + logfile, 'a')

    days2sec = 86400

    # Read the config file into a dictionary
    print("Reading the config file.")
    config = ConfigParser.SafeConfigParser()
    config.read([cfile])
    configdict = dict(config.items("MCMC"))

    # Pull some variables out
    plots = configdict['plots'] == 'True'
    bins  = configdict['bins']  == 'True'

    # Get initial parameters and stepsize arrays from the config
    stepsize = [float(s) for s in configdict['stepsize'].split()]
    params   = [float(s) for s in configdict['params'].split()]

    # Load the POET event object (up through p5)
    print("Loading the POET event object.")
    event_chk = me.loadevent(eventname + "_p5c")
    event_pht = me.loadevent(eventname + "_pht")
    event_ctr = me.loadevent(eventname + "_ctr", load=['data', 'uncd', 'mask'])

    data  = event_ctr.data
    uncd  = event_ctr.uncd
    phase = event_chk.phase[0]


    # Identify the bright pixels to use
    print("Identifying brightest pixels.")
    nx = data.shape[1]
    ny = data.shape[2]
    
    phot    = event_pht.fp.aplev[np.where(event_chk.good)]
    photerr = event_pht.fp.aperr[np.where(event_chk.good)]

    xavg = np.int(np.floor(np.average(event_pht.fp.x)))
    yavg = np.int(np.floor(np.average(event_pht.fp.y)))

    boxsize = 10
    
    photavg     = np.average(data[:,yavg-boxsize:yavg+boxsize,xavg-boxsize:xavg+boxsize], axis=0)[:,:,0]
    photavgflat = photavg.flatten()

    # Some adjustable parameters that should be at the top of the file
    npix = 9
    necl = 6 #number of eclipse parameters

    flatind = photavgflat.argsort()[-npix:]

    rows = flatind / photavg.shape[1]
    cols = flatind % photavg.shape[0]

    pixels = []

    for i in range(npix):
        pixels.append([rows[i]+yavg-boxsize,cols[i]+xavg-boxsize])
    
    # Default to 3x3 box of pixels
    # avgcentx = np.floor(np.average(event_pht.fp.x) + 0.5)
    # avgcenty = np.floor(np.average(event_pht.fp.y) + 0.5)
    # avgcent  = [avgcenty, avgcentx]
    # pixels = []
	   
    # for i in range(3):
    #     for j in range(3):
    #         pixels.append([avgcenty - 1 + i, avgcentx - 1 + j])

    print("Doing preparatory calculations.")
    phat, dP = zf.zen_init(data, pixels)

    phatgood = np.zeros(len(event_chk.good[0]))
    
    # Mask out the bad images in phat
    for i in range(npix):
        tempphat = phat[:,i].copy()
        tempphatgood = tempphat[np.where(event_chk.good[0])]
        if i == 0:
            phatgood = tempphatgood.copy()
        else:
            phatgood = np.vstack((phatgood, tempphatgood))
        del(tempphat)
        del(tempphatgood)
        
    # Invert the new array because I lack foresight
    phatgood  = phatgood.T
    phasegood = event_chk.phase[np.where(event_chk.good)]

    # Do binning if desired
    if bins:
        # Width of bins to try
        bintry = np.array([ 4.,
                            8.,
                           12.,
                           16.,
                           20.,
                           24.,
                           28.,
                           32.,
                           36.,
                           40.,
                           44.,
                           48.,
                           52.,
                           56.,
                           60.,
                           64.])

        #bintry = np.arange(4,129,dtype=float)

        # Convert bin widths to phase from seconds
        bintry /= (event_chk.period * days2sec)

        # Initialize best chi-squared to an insanely large number
        # for comparison later
        chibest = 1e300

        chisqarray = np.zeros(len(bintry))

        # Optimize bin size
        print("Optimizing bin size.")
        for i in range(len(bintry)):
            print("Least-squares optimization for "
                  + str(bintry[i] * event_chk.period * days2sec)
                  + " second bin width.")

            # Bin the phase and phat
            for j in range(npix):
                if j == 0:
                    binphase,     binphat = zf.bindata(phasegood, phatgood[:,j], bintry[i])
                else:
                    binphase, tempbinphat = zf.bindata(phasegood, phatgood[:,j], bintry[i])
                    binphat = np.column_stack((binphat, tempbinphat))
            # Bin the photometry and error
            # Phase is binned again but is identical to
            # the previously binned phase.
            binphase, binphot, binphoterr = zf.bindata(phasegood, phot, bintry[i], yerr=photerr)

            # Normalize
            photnorm    = phot    / phot.mean()
            photerrnorm = photerr / phot.mean()

            binphotnorm    = binphot    / binphot.mean()
            binphoterrnorm = binphoterr / binphot.mean()

            # Make xphat for use with zen_optimize
            xphatshape = (binphat.shape[0], binphat.shape[1]+1)
            xphat      = np.zeros(xphatshape)

            xphat[:,:-1] = binphat
            xphat[:, -1] = binphase

            # Minimize chi-squared for this bin size
            ret = sco.curve_fit(zf.zen_optimize, xphat, binphotnorm, p0=params, sigma=binphoterrnorm, maxfev = 100000)

            # Calculate the best-fitting model
            model = zf.zen(ret[0], binphase, binphat, npix)

            # Calculate reduced chi-squared
            chisq = np.sum((binphotnorm - model)**2/binphoterrnorm**2)
            redchisq = chisq/len(binphotnorm)
            print("Reduced chi-squared: " + str(redchisq))

            chisqarray[i] = redchisq

            # Save results if this fit is better
            if redchisq < chibest:
                chibest = redchisq
                binbest = bintry[i]

        # Rebin back to the best binning
        binphase, binphot, binphoterr = zf.bindata(phasegood, phot, binbest, yerr=photerr)
        binphotnorm    = binphot    / binphot.mean()
        binphoterrnorm = binphoterr / binphot.mean()

        print("Best bin size: " +
              str(binbest * event_chk.period * days2sec) +
              " seconds.")
        lf.write("Best bin size: " +
                 str(binbest * event_chk.period * days2sec) +
                 " seconds.")

        for j in range(npix):
            if j == 0:
                binphase,     binphat = zf.bindata(phasegood, phatgood[:,j], binbest)
            else:
                binphase, tempbinphat = zf.bindata(phasegood, phatgood[:,j], binbest)
                binphat = np.column_stack((binphat, tempbinphat))

        if plots:
            plt.clf()
            plt.plot(bintry * event_chk.period * days2sec, chisqarray)
            plt.xlabel("Bin width (seconds)")
            plt.ylabel("Reduced Chi-squared")
            plt.title("Reduced Chi-squared of PLD model fit for different bin sizes")
            plt.savefig(outdir+"redchisq.png")
            
    # If not binning, use regular photometry
    else:
        photnorm       = phot    / phot.mean()
        photerrnorm    = photerr / phot.mean()
        binphotnorm    = photnorm.copy()
        binphoterrnorm = photerrnorm.copy()
        binphase       = phasegood.copy()
        binphat        = phatgood.copy()

    # And we're off!    
    print("Beginning MCMC.")
    savefile = configdict['savefile']
    log      = configdict['logfile']
    
    bp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(binphotnorm,
                                                       binphoterrnorm,
                                                       func=zf.zen,
                                                       indparams=[binphase,
                                                                  binphat,
                                                                  npix],
                                                       cfile=cfile,
                                                       savefile=outdir+savefile,
                                                       log=outdir+log)


    # Get initial parameters and stepsize arrays from the config
    stepsize = [float(s) for s in configdict['stepsize'].split()]
    params   = [float(s) for s in configdict['params'].split()]

    # Calculate the best-fitting model
    bestfit = zf.zen(bp, binphase, binphat, npix)

    # Get parameter names array to match params with names
    parnames = configdict["parname"].split()

    # Make array of parameters, with eclipse depth replaced with 0
    noeclParams = np.zeros(len(bp))

    for i in range(len(noeclParams)):
        if parnames[i] == 'Depth':
            noeclParams[i] == 0
            depth = bp[i]
        else:
            noeclParams[i] = bp[i]

    noeclfit = zf.zen(noeclParams, binphase, binphat, npix)

    bestecl = depth*(zf.eclipse(binphase, bp[npix:npix+necl])-1) + 1

    # Make plots
    print("Making plots.")
    binnumplot = 200
    binplotwidth = (phasegood[-1]-phasegood[0])/binnumplot
    binphaseplot, binphotplot, binphoterrplot = zf.bindata(binphase, binphot,  binplotwidth, yerr=binphoterr)
    binphaseplot, binnoeclfit                 = zf.bindata(binphase, noeclfit, binplotwidth)
    binphaseplot, binbestecl                  = zf.bindata(binphase, bestecl,  binplotwidth)
    binphotnormplot    = binphotplot    / binphotplot.mean()
    binphoterrnormplot = binphoterrplot / binphotplot.mean()
    zp.normlc(binphaseplot[:-1], binphotnormplot[:-1], binphoterrnormplot[:-1],
              binnoeclfit[:-1], binbestecl[:-1], 1,
              title='Normalized Binned WASP-29b Data With Eclipse Models', savedir=outdir)

if __name__ == "__main__":
    main()
