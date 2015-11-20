#em 2 nov 2015

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

# Load the POET event object (up through p5)
event = me.loadevent(event.eventname + "_pht")

# Need to load a config for MC3 param priors

# Need to load a config for pixels to use
# Probably should be relative to average pixel center from centering.
# Perhaps some default shapes? Like a box, plus-shape, or circle of radius r


# FINDME: where are images stored in the event object?
# data = event.
# pixels = 

phat, dP = zf.zen_init(data, pixels)

npix = len(pixels)

# FINDME: This is the general structure we need for MC3, but names/numbers are subject to change
# FINDME: Should read a config for params, numit, burnin
allp, bp = mc3.mcmc(data, uncert, func=zf.zen, indparams=[time, phat, npix],
                    params=params, numit=1e5, burnin=500)

