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
sys.path.insert(2, "./poetlib")
import MCcubed as mc3
import manageevent as me
import multiprocessing as mp

def main():
    '''
    One function to rule them all.
    '''

    print("Start: %s" % time.ctime())
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

    logfile = 'zen.log'
    lf = open(outdir + logfile, 'a')

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

    # Get slope threshold
    slopethresh = float(configdict['slopethresh'])

    # Get number of binning processes
    nprocbin    = int(configdict['nprocbin'])

    # Get clip
    preclip  = float(configdict['preclip'])
    postclip = float(configdict['postclip'])

    # Width of bins to try (points per bin)
    bintry = np.arange(-2, 259, 4)
    bintry[0] = 1

    # Initialize array of chi-squared. This will hold the chi-squared valued
    # of the binned residuals compared with a line with slope -0.5. See
    # Deming et al. 2015
    #chisqarray = np.zeros((len(bintry), len(photdir), len(centdir)))

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
                # Initialize process
                p = mp.Process(target = zf.do_bin,
                               args = (bintry, phasegood, phatgood, phot,
                                       photerr, params, npix, stepsize,
                                       pmin, pmax, chisqarray, chislope,
                                       l, k, len(photdir)))

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
    if np.all(chislope) > slopethresh:
        print("Slope threshold too low. Increase and rerun.")
        print("Setting threshhold to 0.")
        slopethresh = 0.0

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
    
    print("Reloading best POET object.")
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

    print("Rebinning to the best binning.")
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

    print("End:  %s" % time.ctime())
if __name__ == "__main__":
    main()
