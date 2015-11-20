#em 2 nov 2015

import sys
import numpy as np
import zen_funcs as zf
import matplotlib.pyplot as plt
sys.path.appen("./MCcubed/src/")
import mccubed as mc3


#somehow get data here, either read in something or unpack the poet event
#also a config file of the initial parameter values or whatever for mcmc

phat, dP = zf.zen_init(data, pixels)

# FINDME: This is the general structure we need for MC3, but names/numbers are subject to change
allp, bp = mc3.mcmc(data, uncert, func=zf.zen, indparams=time,
                    params=params, numit=1e5, burnin=500)

