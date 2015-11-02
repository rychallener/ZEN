#em 2 nov 2015

import numpy as np
import zen_funcs as zf
import matplotlib.pyplot as plt
import mccubed as mc3


#somehow get data here, either read in something or unpack the poet event
#also a config file of the initial parameter values or whatever for mcmc

phat, dP = zf.zen_init(values)

#call mc3.mcmc with zen as the function
