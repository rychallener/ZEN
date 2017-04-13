#! /usr/bin/env python2

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
#   2017-02-20 rchallen Lots of bug fixes. Made git repo.
# See git repo for full history.

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
    plots  = configdict['plots']  == 'True'
    bins   = configdict['bins']   == 'True'
    titles = configdict['titles'] == 'True'

    # Get apertures to try
    poetdir = configdict['poetdir']
    centdir = [s for s in configdict['cent'].split()]
    photdir = [s for s in configdict['phot'].split()]
    
    # Get initial parameters and stepsize arrays from the config
    stepsize = [float(s) for s in configdict['stepsize'].split()]
    params   = [float(s) for s in configdict['params'  ].split()]
    pmin     = [float(s) for s in configdict['pmin'    ].split()]
    pmax     = [float(s) for s in configdict['pmax'    ].split()]

    # Get number of pixels to use from the config
    npix     = int(configdict['npix'])

    # Get preclip
    preclip  = float(configdict['preclip'])
    postclip = float(configdict['postclip'])

    # Width of bins to try (points per bin)
    bintry = np.arange(-2, 259, 4)
    bintry[0] = 1

    # Initialize array of chi-squared. This will hold the chi-squared valued
    # of the binned residuals compared with a line with slope -0.5. See
    # Deming et al. 2015
    chisqarray = np.zeros((len(bintry), len(photdir), len(centdir)))

    # Giant loop over all specified apertures and centering methods
    for l in range(len(photdir)):
        for k in range(len(centdir)):            
            # Load the POET event object (up through p5)
            print("Loading the POET event object.")
            print("Ap:   " + photdir[l])
            print("Cent: " + centdir[k])
            centloc = poetdir + centdir[k] + '/' 
            photloc = poetdir + centdir[k] + '/' + photdir[l] + '/'
            event_chk = me.loadevent(photloc + eventname + "_p5c")
            event_pht = me.loadevent(photloc + eventname + "_pht")
            event_ctr = me.loadevent(centloc + eventname + "_ctr", load=['data', 'uncd', 'mask'])

            data  = event_ctr.data
            uncd  = event_ctr.uncd
            phase = event_chk.phase

            preclipmask  = phase >  preclip
            postclipmask = phase < postclip
            clipmask = np.logical_and(preclipmask, postclipmask)
            mask     = np.logical_and(   clipmask, event_chk.good)

            # Identify the bright pixels to use
            print("Identifying brightest pixels.")
            nx = data.shape[1]
            ny = data.shape[2]

            phot    = event_pht.fp.aplev[mask]
            photerr = event_pht.fp.aperr[mask]

            xavg = np.int(np.floor(np.average(event_pht.fp.x)))
            yavg = np.int(np.floor(np.average(event_pht.fp.y)))

            boxsize = 10

            photavg     = np.average(data[:,yavg-boxsize:yavg+boxsize,
                                            xavg-boxsize:xavg+boxsize], axis=0)[:,:,0]
            photavgflat = photavg.flatten()

            # Some adjustable parameters that should be at the top of the file
            necl = 6 #number of eclipse parameters

            flatind = photavgflat.argsort()[-npix:]

            rows = flatind / photavg.shape[1]
            cols = flatind % photavg.shape[0]

            pixels = []

            for i in range(npix):
                pixels.append([rows[i]+yavg-boxsize,cols[i]+xavg-boxsize])

            print("Doing preparatory calculations.")
            phat, dP = zf.zen_init(data, pixels)

            phatgood = np.zeros(len(mask))

            # Mask out the bad images in phat
            for i in range(npix):
                tempphat = phat[:,i].copy()
                tempphatgood = tempphat[mask[0]]
                if i == 0:
                    phatgood = tempphatgood.copy()
                else:
                    phatgood = np.vstack((phatgood, tempphatgood))
                    del(tempphat)
                    del(tempphatgood)

            # Invert the new array because I lack foresight
            phatgood  = phatgood.T
            phasegood = event_chk.phase[mask]

            # Do binning if desired
            if bins:

                # Optimize bin size
                print("Optimizing bin size.")
                for i in range(len(bintry)):
                    print("Least-squares optimization for " + str(bintry[i])
                          + " points per bin.")

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

                    # Minimize chi-squared for this bin size
                    indparams = [binphase, binphat, npix]
                    chisq, fitbestp, dummy, dummy = mc3.fit.modelfit(params, zf.zen,
                                                                     binphotnorm,
                                                                     binphoterrnorm,
                                                                     indparams,
                                                                     stepsize,
                                                                     pmin, pmax)

                    # Calculate model on unbinned data from parameters of the
                    # chi-squared minimization with this bin size. Calculate
                    # residuals for binning
                    sdnr        = []
                    binlevel    = []
                    err         = []
                    resppb      = 1
                    unbinnedres = photnorm - zf.zen(fitbestp, phasegood, phatgood, npix)
                    resbin = len(phasegood)

                    # Bin the residuals, calculate SDNR of binned residuals. Do this
                    # until you are binning to <= 16 points remaining.
                    while resbin > 16:
                        dummy, binnedres = zf.bindata(phasegood, unbinnedres, resppb)
                        sdnr.append(np.std(binnedres))
                        binlevel.append(resppb)
                        err.append(np.std(binnedres)/(2.0*len(binnedres)))
                        resppb *= 2
                        resbin = int(resbin / 2)

                    # Calculate chisquared of the various SDNR wrt line of slope -0.5
                    # passing through the SDNR of the unbinned residuals
                    # Record chisquared
                    sdnr     = np.asarray(sdnr)
                    binlevel = np.asarray(binlevel)
                    sdnrchisq = zf.reschisq(sdnr, binlevel, err)
                    chisqarray[i,l,k] = sdnrchisq

                if plots:
                    plt.clf()
                    plt.plot(bintry, chisqarray[:,l,k])
                    plt.xlabel("Bin width (ppbin)")
                    plt.ylabel("Chi-squared")
                    if titles:
                        plt.title("Chi-squared of log(SDNR) vs. log(bin width) compared to theory")
                        plt.savefig(outdir+photdir[l]+"-"+centdir[k]+"-redchisq.png")

            # If not binning, use regular photometry
            else:
                photnorm       = phot    / phot.mean()
                photerrnorm    = photerr / phot.mean()
                binphotnorm    = photnorm.copy()
                binphoterrnorm = photerrnorm.copy()
                binphase       = phasegood.copy()
                binphat        = phatgood.copy()

    # Determine best binning
    bestindex = np.where(chisqarray == np.min(chisqarray))

    ibin  = bestindex[0][0]
    iphot = bestindex[1][0]
    icent = bestindex[2][0]
    print(ibin)
    print(iphot)
    print(icent)

    print("Best aperture:  " +     photdir[iphot])
    print("Best centering: " +     centdir[icent])
    print("Best binning:   " + str(bintry[ibin]))

    lf.write("Best aperture:  " +     photdir[iphot])
    lf.write("Best centering: " +     centdir[icent])
    lf.write("Best binning:   " + str(bintry[ibin]))
    
    # Reload the event object
    # We don't load them all in at once because they each use a
    # considerable amount of memory, and one could conceivably
    # want to check dozens of combinations of centering and
    # photometry radii/methods
    centloc = poetdir + centdir[icent] + '/'
    photloc = poetdir + centdir[icent] + '/' + photdir[iphot] + '/'
    event_chk = me.loadevent(photloc + eventname + "_p5c")
    event_pht = me.loadevent(photloc + eventname + "_pht")
    event_ctr = me.loadevent(centloc + eventname + "_ctr", load=['data', 'uncd', 'mask'])

    binbest = bintry[ibin]
    
    binphase, binphot, binphoterr = zf.bindata(phasegood, phot, binbest, yerr=photerr)
    binphotnorm    = binphot    / binphot.mean()
    binphoterrnorm = binphoterr / binphot.mean()

    for j in range(npix):
        if j == 0:
            binphase,     binphat = zf.bindata(phasegood, phatgood[:,j], binbest)
        else:
            binphase, tempbinphat = zf.bindata(phasegood, phatgood[:,j], binbest)
            binphat = np.column_stack((binphat, tempbinphat))
                
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
    binnumplot = int(len(binphot)/60)
    binphaseplot, binphotplot, binphoterrplot = zf.bindata(binphase,
                                                           binphot,
                                                           binnumplot,
                                                           yerr=binphoterr)
    binphaseplot, binnoeclfit                 = zf.bindata(binphase,
                                                           noeclfit,
                                                           binnumplot)
    binphaseplot, binbestecl                  = zf.bindata(binphase,
                                                           bestecl,
                                                           binnumplot)
    binphotnormplot    = binphotplot    / binphotplot.mean()
    binphoterrnormplot = binphoterrplot / binphotplot.mean()
    
    if titles:
        zp.normlc(binphaseplot[:-1], binphotnormplot[:-1],
                  binphoterrnormplot[:-1], binnoeclfit[:-1],
                  binbestecl[:-1], 1,
                  title='Normalized Binned WASP-29b Data With Eclipse Models',
                  savedir=outdir)
    else:
        zp.normlc(binphaseplot[:-1], binphotnormplot[:-1],
                  binphoterrnormplot[:-1], binnoeclfit[:-1],
                  binbestecl[:-1], 1,
                  title='',
                  savedir=outdir)

if __name__ == "__main__":
    main()
