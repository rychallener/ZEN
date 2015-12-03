#em 2 nov 2015

import numpy as np

def zen_init(data, pixels):
	"""
	This function does the initial calculations for pixel-level decorrelation.

	Parameters:
	-----------
	data: ndarray
	  3D float array of images

	pixels: ndarray
	  2D array coordinates of pixels to consider
	  EX: array([[ 0,  1],
				 [ 2,  3],
				 [ 4,  5],
				 [ 6,  7],
				 [ 8,  9],
				 [10, 11],
				 [12, 13],
				 [14, 15],
				 [16, 17],
				 [18, 19]])

	Returns:
	--------

	Example:
	--------
	>>> import numpy as np
	>>> data = np.arange(200).reshape(8,5,5)
	>>> pixels = [[1,2],[2,1],[2,2],[2,3],[3,2]]
	>>> res = zen_init(data,pixels)

	Modification History:
	---------------------
	2015-11-02 em	   Initial implementation
	2015-11-20 rchallen Generalized for any given pixels

d	"""
	# Set number of frames and image dimensions
	nframes, ny, nx, nsets = np.shape(data)

	# Set number of pixels
	npix = len(pixels)
	
	# Initialize array for flux values of pixels
	p = np.zeros((nframes, npix))

	# Extract flux values from images
	for i in range(npix):
		for j in range(nframes):
			p[j,i] = data[j, pixels[i][0], pixels[i][1],0]

	# Initialize array for normalized flux of pixels
	phat = np.zeros(p.shape)
	
	# Remove astrophysics by normalizing
	for t in range(nframes):
		phat[t]	= p[t]/np.sum(p[t])

	# Calculate mean flux through all frames at each pixel
	pbar	= np.mean(phat, axis=0)

	# Difference from the mean flux
	dP	= phat - pbar

	return(phat, dP)

def eclipse(t, eclparams):
	"""
	This function calculates an eclipse following Mandel & Agol (2002)
	Adapted from mandelecl.c used in p6.

	Modification History
	--------------------
	2015-11-30 rchallen Adapted from C in mandelecl.c
	"""

	# Set parameter names
	midpt = eclparams[0]
	width = eclparams[1]
	depth = eclparams[2]
	t12   = eclparams[3]
	t34   = eclparams[4]
	flux  = eclparams[5]

	# If zero depth, set model to a flat line at 1
	if depth == 0:
		y = np.ones(t.size)
		return y

	# Beginning of eclipse
	t1 = midpt - width/2

	# End of eclipse
	if t1 + t12 < midpt:
		t2 = t1+t12
	else:
		t2 = midpt

	t4 = midpt + width / 2

	if t4 - t34 > midpt:
		t3 = t4 - t34
	else:
		t3 = midpt

	p = np.sqrt(np.abs(depth)) * (depth/np.abs(depth))

	dims = t.size
	
	y = np.ones(dims)

	# Calculate ingress/egress
	for i in range(dims):
		y[i] = 1
		if t[i] >= t2 and t[i] <= t3:
			y[i] = 1 - depth

		elif p != 0:
			if t[i] >= t1 and t[i] <= t2:
				z  = -2 * p * (t[i] - t1) / t12 + 1 + p
				k0 = np.arccos((p**2 + z**2 - 1) / (2 * p * z))
				k1 = np.arccos((1 - p**2 + z**2) / (2 * z))
				y[i] = 1 - depth/np.abs(depth)/np.pi * \
				  (p**2 * k0 + k1 - np.sqrt((4 * z**2 - (1 + z**2 - p**2)**2)/4))

			elif t[i] > t3 and t[i] < t4:
				z  = 2 * p * (t[i] - t3) / t34 + 1 - p
				k0 = np.arccos((p**2 + z**2 - 1) / (2 * p * z))
				k1 = np.arccos((1 - p**2 + z**2) / (2 * z))
				y[i] = 1 - depth/np.abs(depth)/np.pi * \
				  (p**2 * k0 + k1 - np.sqrt((4 * z**2 - (1 + z**2 - p**2)**2)/4))

		#y[i] *= flux

	return y

def zen(par, x, phat, npix):
	"""
	Zen function.

	Parameters:
	-----------
	par: ndarray
	  Zen parameters, eclipse parameters, ramp parameters
	x: ndarray
	  Locations to evaluate eclipse model
	npix: int
	  

	Returns:
	--------
	y: ndarray
	  Model evaluated at x

	Modification History:
	---------------------
	2015-11-02 em	   Initial implementation.
	2015-11-20 rchallen Adaptation for use with MCcubed

	Notes:
	------
	Only allows for quadratic ramp functions
	"""

	# PLD term in the model
	PLD = 0
	
	# Calculate eclipse model using parameters
	# FINDME: remove hard-coded number of ramp params
	eclparams = par[npix:-3]
	eclmodel = eclipse(x, eclparams)
	
	# Calculate the sum of the flux from all considered pixels
	for i in range(npix):
		PLD += par[i]*phat[:,i]

	# Calculate the model
	# FINDME: allow for general ramp function
	#y = ((1 + PLD) + (eclmodel - 1) + (par[-3] + par[-2]*x + par[-1]*x**2))*eclparams[-1]

	y = eclparams[-1] * (1 + PLD + (eclmodel - 1) + (par[-3] + par[-2]*x + par[-1]*x**2))
	return y









