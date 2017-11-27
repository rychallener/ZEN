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
import shutil
import numpy as np
import scipy.optimize as sco
import zen_funcs as zf
import zenplots as zp
import matplotlib.pyplot as plt
import ConfigParser
sys.path.insert(1, "./mccubed/")
sys.path.insert(2, "./lib")
import MCcubed as mc3
import manageevent as me
import multiprocessing as mp
import logger
import kurucz_inten
import irsa
import copy
import readeventhdf
from sklearn import linear_model

# This forces the code to use the version
# of kurucz_inten within the ./lib directory.
# Must be a better way to do so.
reload(kurucz_inten)

def main():
    '''
    One function to rule them all.
    '''

    pld = False
    regress = False
    
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
    
    shutil.copy2(cfile, outdir + cfile)

    # Set up logging of all print statements in this main file
    logfile = outdir + 'zen.log'
    sys.stdout = logger.Logger(logfile)        
    print("Start: %s" % time.ctime())
    
    shutil.copy2(cfile, outdir + cfile)

    # Set up logging of all print statements in this main file
    logfile = outdir + 'zen.log'
    sys.stdout = logger.Logger(logfile)

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

    # Get parameter names array to match params with names
    parnames = configdict["parname"].split()

    # Get number of pixels to use from the config
    npix     = int(configdict['npix'])

    # Get slope threshold
    slopethresh = float(configdict['slopethresh'])

    # Get number of binning processes
    nprocbin    = int(configdict['nprocbin'])

    # Get clip
    preclip  = float(configdict['preclip'])
    postclip = float(configdict['postclip'])

    # Width of bins to try (points per bin)
    bintry = [int(s) for s in configdict['bintry'].split()]

    # Make a default for bins to try
    if bintry == [0]:
      bintry = np.arange(-2, 259, 4)
      bintry[0] = 1

    # Set up multiprocessing
    jobs = []
    # Multiprocessing requires 1D arrays (if we use shared memory)
    chisqarray = mp.Array('d', np.zeros(len(bintry)  *
                                        len(photdir) *
                                        len(centdir)))
    chislope   = mp.Array('d', np.zeros(len(bintry)  *
                                        len(photdir) *
                                        len(centdir)))

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
            event_ctr = me.loadevent(centloc + eventname + "_ctr",
                                     load=['data', 'uncd', 'mask'])

            data  = event_ctr.data
            uncd  = event_ctr.uncd
            phase = event_chk.phase

            preclipmask  = phase >  preclip
            postclipmask = phase < postclip
            clipmask = np.logical_and(preclipmask, postclipmask)
            mask     = np.logical_and(   clipmask, event_chk.good)

            phasegood = event_chk.phase[mask]

            # Identify the bright pixels to use
            print("Identifying brightest pixels.")
            nx = data.shape[1]
            ny = data.shape[2]

            phot    = event_pht.fp.aplev[mask]
            photerr = event_pht.fp.aperr[mask]

            normfactor = np.average(phot)
            
            phot    /= normfactor
            photerr /= normfactor

            xavg = np.int(np.floor(np.average(event_pht.fp.x)))
            yavg = np.int(np.floor(np.average(event_pht.fp.y)))

            boxsize = 10

            photavg = np.average(data[:,yavg-boxsize:yavg+boxsize,
                                        xavg-boxsize:xavg+boxsize],
                                 axis=0)[:,:,0]
            
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

            if pld:
                phatgood = np.loadtxt('phat.txt', delimiter=',')
                phasegood = np.loadtxt('phase.txt')
                phot = np.loadtxt('phot.txt')
                photerr = np.loadtxt('err.txt')

            # Here we estimate the midpoint of the eclipse, which is
            # necessary if using linear regression. Note that this
            # updates the params array
            if regress:
                # This section has some hard-coded numbers
                # which should be removed, and put into
                # a config. Although I intend to remove
                # the regression option eventually.
                print("Estimating midpoint")
                for m in range(len(trymid)):
                    midpt = params[12]
                
                    trymid = np.linspace(midpt - 0.01, midpt + 0.01, 100)

                    chibest = 1e300
                    
                    clf = linear_model.LinearRegression(fit_intercept=False)

                    eclpars = np.asarray(params[npix:-3])
                    eclpars[0] = trymid[m]
                    eclmodel = zf.eclipse(phasegood, eclpars)

                    xx = np.append(phatgood,
                                   np.reshape(eclmodel,  (len(phasegood),1)),
                                   axis=1)
                    xx = np.append(xx,
                                   np.reshape(phasegood, (len(phasegood),1)),
                                   axis=1)
                    xx = np.append(xx,
                                   np.ones((len(phasegood),1)),
                                   axis=1)

                    clf.fit(xx, phot)

                    model = np.sum(clf.coef_*xx,axis=1)

                    chitry = np.sum((phot - model)**2/photerr**2)

                    if chitry < chibest:
                        chibest = chitry
                        midbest = trymid[m]

                print("Midpoint guess: " + str(midbest))
                params[12] = midbest
            # Do binning if desired
            if bins:
                # Optimize bin size                
                # Initialize processes
                p = mp.Process(target = zf.do_bin,
                               args = (bintry, phasegood, phatgood, phot,
                                       photerr, params, npix, stepsize,
                                       pmin, pmax, chisqarray, chislope,
                                       l, k, len(photdir),
                                       regress))

                # Start process
                jobs.append(p)
                p.start()

                # This intentionally-infinite loop continuously calculates
                # the number of running processes, then exits if the number
                # of processes is less than the number requested. This allows
                # additional processes to spawn as other finish, which is more
                # efficient than waiting for them all to finish since some
                # processes can take much longer than others
                while True:
                    procs = 0
                    for proc in jobs:
                        if proc.is_alive():
                            procs += 1

                    if procs < nprocbin:
                        break
                    
                    # Save the CPU some work.
                    time.sleep(0.1)
                    
                # Move this to a separate loop eventually
                # if plots:
                #     plt.clf()
                #     plt.plot(bintry, chisqarray[:,l,k])
                #     plt.xlabel("Bin width (ppbin)")
                #     plt.ylabel("Chi-squared")
                #     if titles:
                #         plt.title("Chi-squared of log(SDNR) vs. log(bin width) compared to theory")
                #     plt.savefig(outdir+photdir[l]+"-"+centdir[k]+"-redchisq.png")

            # If not binning, use regular photometry
            else:
                photnorm       = phot    / phot.mean()
                photerrnorm    = photerr / phot.mean()
                binphotnorm    = photnorm.copy()
                binphoterrnorm = photerrnorm.copy()
                binphase       = phasegood.copy()
                binphat        = phatgood.copy()

    # Make sure all processes finish
    for proc in jobs:
        proc.join()

    chisqarray = np.asarray(chisqarray).reshape((len(centdir),
                                                 len(photdir),
                                                 len(bintry)))
    chislope   = np.asarray(chislope  ).reshape((len(centdir),
                                                 len(photdir),
                                                 len(bintry)))

    # Initialize chibest to something ridiculous
    chibest = 1e100
    # Determine best binning
    # We also demand that the slope be less than a
    # value, because Deming does and if the slope is
    # too far off from -1/2, binning is not improving the
    # fit in a sensible way
    if all(i >= slopethresh for i in chislope.flatten()):
        print("Slope threshold too low. Increase and rerun.")
        print("Setting threshold to 0 so run can complete.")
        slopethresh = 0
                
                
    for i in range(len(centdir)):
        for j in range(len(photdir)):
            for k in range(len(bintry)):
                if chisqarray[i,j,k] < chibest and chislope[i,j,k] < slopethresh:
                    chibest   = chisqarray[i,j,k]
                    slopebest = chislope  [i,j,k]
                    icent = i
                    iphot = j
                    ibin  = k

    print("Best aperture:  " +     photdir[iphot])
    print("Best centering: " +     centdir[icent])
    print("Best binning:   " + str(bintry[ibin]))
    print("Slope of SDNR vs Bin Size: " + str(slopebest))
    
    # Reload the event object
    # We don't load them all in at once because they each use a
    # considerable amount of memory, and one could conceivably
    # want to check dozens of combinations of centering and
    # photometry radii/methods
    centloc = poetdir + centdir[icent] + '/'
    photloc = poetdir + centdir[icent] + '/' + photdir[iphot] + '/'
    
    print("Reloading best POET object.")
    event_chk = me.loadevent(photloc + eventname + "_p5c")
    event_pht = me.loadevent(photloc + eventname + "_pht")
    event_ctr = me.loadevent(centloc + eventname + "_ctr",
                             load=['data', 'uncd', 'mask'])

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

    photavg     = np.average(data[:,yavg-boxsize:yavg+boxsize,
                                    xavg-boxsize:xavg+boxsize], axis=0)[:,:,0]
    photavgflat = photavg.flatten()

    flatind = photavgflat.argsort()[-npix:]

    rows = flatind / photavg.shape[1]
    cols = flatind % photavg.shape[0]

    pixels = []

    for i in range(npix):
        pixels.append([rows[i]+yavg-boxsize,cols[i]+xavg-boxsize])

    print("Redoing preparatory calculations.")
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

    if pld:
        phatgood = np.loadtxt('phat.txt', delimiter=',')
        phasegood = np.loadtxt('phase.txt')
        phot = np.loadtxt('phot.txt')
        photerr = np.loadtxt('err.txt')
    
    print("Rebinning to the best binning.")
    binbest = bintry[ibin]
    
    binphase, binphot, binphoterr = zf.bindata(phasegood, phot, binbest, yerr=photerr)
    binphotnorm    = binphot    / phot.mean()
    binphoterrnorm = binphoterr / phot.mean()

    for j in range(npix):
        if j == 0:
            binphase,     binphat = zf.bindata(phasegood,
                                               phatgood[:,j],
                                               binbest)
        else:
            binphase, tempbinphat = zf.bindata(phasegood,
                                               phatgood[:,j],
                                               binbest)
            binphat = np.column_stack((binphat, tempbinphat))

    
    # And we're off!    
    print("Beginning MCMC.")
    savefile = configdict['savefile']
    log      = configdict['logfile']

    if regress == False:
        bp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(binphotnorm,
                                                       binphoterrnorm,
                                                       func=zf.zen,
                                                       indparams=[binphase,
                                                                  binphat,
                                                                  npix],
                                                       cfile=cfile,
                                                       savefile=outdir+savefile,
                                                       log=outdir+log)

        bpres   = binphotnorm - zf.zen(bp, binphase, binphat, npix)
    else:
        #eclmodel = np.loadtxt('pldecl.txt')
        #binecl, dummy = zf.bindata(eclmodel, eclmodel, binbest)
        eclmodel = zf.eclipse(binphase, params[npix:-3])
        #binecl, dummy = zf.bindata(eclmodel, eclmodel, binbest)

        xx = np.append(binphat,
                       np.reshape(eclmodel, (len(binphase),1)),
                       axis=1)
        xx = np.append(xx,
                       np.ones((len(binphase),1)),
                       axis=1)
        xx = np.append(xx,
                       np.reshape(binphase, (len(binphase),1)),
                       axis=1)

        bp, CRlo, CRhi, stdp, posterior, Zchain = mc3.mcmc(binphotnorm,
                                                           binphoterrnorm,
                                                           func=zf.zen2,
                                                           indparams=[xx],
                                                           cfile=cfile,
                                                           savefile=outdir+savefile,
                                                           log=outdir+log)
        bpres = binphotnorm - zf.zen2(bp, xx)
     
    bpchisq = np.sum(bpres**2/binphoterrnorm**2)

    nfreep = int(np.sum(stepsize > 0))
    ndata  = len(binphotnorm)

    chifactor = np.sqrt(bpchisq/(ndata - nfreep))
    
    scchisq = np.sum(bpres**2/(binphoterrnorm*chifactor)**2)
    scredchisq = scchisq/(ndata - nfreep)

    print('Best chi-squared:        ' + str(bpchisq))
    print('Scaling factor:          ' + str(chifactor))
    print('Scaled chi-squared:      ' + str(scchisq))
    print('Scaled red. chi-squared: ' + str(scredchisq))

    # Get parameter names array to match params with names
    parnames = configdict["parname"].split()

    # Make array of parameters, with eclipse depth replaced with 0
    noeclParams = np.zeros(len(bp))

    for i in range(len(noeclParams)):
        if parnames[i] == 'Depth':
            noeclParams[i] = 0
            depth = bp[i]
        else:
            noeclParams[i] = bp[i]

    noeclfit = zf.zen(noeclParams, binphase, binphat, npix)
    #noeclfit = zf.zen2(noeclParams, xx)

    bestecl = depth*(zf.eclipse(binphase, bp[npix:npix+necl])-1) + 1

    # Make plots
    print("Making plots.")
    binnumplot = int(len(binphot)/60)
    
    if binnumplot == 0:
        binnumplot = 1
        
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

    # Calculate eclipse times in BJD_UTC and BJD_TDB
    # Code adapted from POET p7
    print('Calculating eclipse times in Julian days')
    offset = event_chk.bjdtdb.flat[0] - event_chk.bjdutc.flat[0]
    if   event_chk.timestd == 'utc':
        ephtimeutc = event_chk.ephtime
        ephtimetdb = event_chk.ephtime + offset
    elif event_chk.timestd == 'tdb':
        ephtimetdb = event_chk.ephtime
        ephtimeutc = event_chk.ephtime - offset
    else:
        print('Assuming that ephemeris is reported in BJD_UTC. Verify!')
        ephtimeutc = event_chk.ephtime
        ephtimetdb = event_chk.ephtime + offset

    print('BJD_TDB - BJD_UTC = ' + str(offset * 86400.) + ' seconds.')

    bestmidpt  =    bp[parnames.index('Midpt')]
    ecltimeerr =  stdp[parnames.index('Midpt')] * event_chk.period

    startutc = event_chk.bjdutc.flat[0]
    starttdb = event_chk.bjdtdb.flat[0]
    
    ecltimeutc = (np.floor((startutc-ephtimeutc)/event_chk.period) +
                  bestmidpt) * event_chk.period + ephtimeutc
    ecltimetdb = (np.floor((starttdb-ephtimetdb)/event_chk.period) +
                  bestmidpt) * event_chk.period + ephtimetdb

    print('Eclipse time = ' + str(ecltimeutc)
          + '+/-' + str(ecltimeerr) + ' BJD_UTC')
    print('Eclipse time = ' + str(ecltimetdb)
          + '+/-' + str(ecltimeerr) + ' BJD_TDB')

    # Brightness temperature calculation
    print('Starting Monte-Carlo Temperature Calculation')
    numcalc = int(configdict['numcalc'])
    
    kout = kurucz_inten.read(event_chk.kuruczfile, freq=True)

    filterf = np.loadtxt(event_chk.filtfile, unpack=True)
    filterf = np.concatenate((filterf[0:2,::-1].T,[filterf[0:2,0]]))
    
    logg     = np.log10(event_chk.tep.g.val*100.)
    loggerr  = np.log10(event_chk.tep.g.uncert*100.)
    tstar    = event_chk.tstar
    tstarerr = event_chk.tstarerr

    # Find index of depth
    countfix = 0
    for i in range(len(parnames)):
        if parnames[i] == 'Depth':
            idepth = i

    # Count number of fixed parameters prior to the depth
    # parameter, to adjust the idepth
    for i in range(idepth):
        if stepsize[i] <= 0:
            countfix += 1

    idepthpost = idepth - countfix
    
    depthpost = posterior[:,idepthpost]
    
    bsdata    = np.zeros((3,numcalc))
    bsdata[0] = depthpost[::posterior.shape[0]/numcalc]
    bsdata[1] = np.random.normal(logg,  loggerr,  numcalc)
    bsdata[2] = np.random.normal(tstar, tstarerr, numcalc)

    tb, tbg, numnegf, fmfreq = zf.calcTb(bsdata, kout, filterf, event_chk)

    tbm   = np.median(tb [np.where(tb  > 0)])
    tbsd  = np.std(   tb [np.where(tb  > 0)])
    tbgm  = np.median(tbg[np.where(tbg > 0)])
    tbgsd = np.std(   tbg[np.where(tbg > 0)])

    print('Band-center brightness temp = '
          + str(round(tbgm,  2)) + ' +/- '
          + str(round(tbgsd, 2)) + ' K')
    print('Integral    brightness temp = '
          + str(round(tbm,  2)) + ' +/- '
          + str(round(tbsd, 2)) + ' K')

    # Make a new event object
    event = copy.copy(event_chk)
    
    # Need to populate the event.fit object with at least the
    # minimum needed to use the do_irsa function
    event.fit = readeventhdf.fits()

    event.fit.fluxuc   = event.fp.aplev[np.where(event.good)] # Unclipped flux
    event.fit.clipmask = clipmask             # Mask for clipping
    event.fit.flux     = event.fp.aplev[mask] # Clipped flux
    event.fit.bestfit  = zf.zen(bp, binphase, binphat, npix) # Best fit (norm)

    # Write IRSA table and FITS file
    os.mkdir(outdir + 'irsa')
    irsa.do_irsa(event, event.fit, directory=outdir)
    
    print("End:  %s" % time.ctime())

if __name__ == "__main__":
    main()
