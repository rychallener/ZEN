#! /usr/bin/env python

# Implementation of PLD algorithm
# Team members:
#   Ryan Challener (rchallen@knights.ucf.edu)
#   Em DeLarme
#   Andrew Foster
#   Chris Macomber
#
# History:
#   2015-11-02 em       Initial implementation
#   2015-11-20 rchallen Updates for use with POET

import sys
import numpy as np
import zen_funcs as zf
import matplotlib.pyplot as plt
sys.path.append("./MCcubed/src/")
sys.path.append("./poetlib")
import mccubed as mc3
import manageevent as me

def main():
    eventname = sys.argv[1]
    cfile     = sys.argv[2]
    
    # Load the POET event object (up through p5)
    event_chk = me.loadevent(eventname + "_chk")
    event_pht = me.loadevent(eventname + "_pht")
    event_ctr = me.loadevent(eventname + "_ctr", load=['data', 'uncd'])

    data  = event_ctr.data
    uncd  = event_ctr.uncd
    phase = event_chk.phase[0]

    phot    = event_pht.fp.aplev[0]
    photerr = event_pht.fp.aperr[0]
    
    # Default to 3x3 box of pixels
    avgcentx = np.floor(np.average(event_pht.fp.x))
    avgcenty = np.floor(np.average(event_pht.fp.y))
    avgcent  = [avgcenty, avgcentx]
    pixels = []
	   
    for i in range(3):
        for j in range(3):
            pixels.append([avgcenty - 1 + i, avgcentx - 1 + j])
		  
    phat, dP = zf.zen_init(data, pixels)

    npix  = len(pixels)
    necl  = 6

    # FINDME: This is the general structure we need for MC3, but names/numbers
    # are subject to change
    allp, bp = mc3.mcmc(phot, photerr, func=zf.zen,
                        indparams=[phase, phat, npix], cfile=cfile)

	return

