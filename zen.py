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
sys.path.appen("./MCcubed/src/")
import mccubed as mc3
import manageevent as me

def main(eventname, cfile):
  # Load the POET event object (up through p5)
  event_pht = me.loadevent(eventname + "_pht")
  event_ctr = me.loadevent(eventname + "_ctr", load=['data'])

  data  = event_ctr.data
  uncd  = event_ctr.uncd
  phase = event_ctr.phase

  phot    = event_pht.fp.aplev
  photerr = event_pht.fp.aperr
  
  # Default to 3x3 box of pixels
  if pixels = None:
      avgcentx = np.floor(np.average(event.fp.x))
      avgcenty = np.floor(np.average(event.fp.y))
      avgcent  = [avgcenty, avgcentx]
      pixels = []
      for i in range(3):
          for j in range(3):
              pixels.append([avgcenty - 1 + i, avgcentx - 1 + j])

  phat, dP = zf.zen_init(data, pixels)

  npix  = len(pixels)
  necl  = 6 #number of eclipse parameters

  # FINDME: This is the general structure we need for MC3, but names/numbers are subject to change
  allp, bp = mc3.mcmc(phot, photerr, func=zf.zen, indparams=[phase, phat, npix],
                      cfile=cfile)

  return

