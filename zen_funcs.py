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
    -------
    >>> import numpy as np
    >>> data = np.arange(200).reshape(8,5,5)
    >>> pixels = [[1,2],[2,1],[2,2],[2,3],[3,2]]
    >>> res = zen_init(data,pixels)

    Modification History:
    ---------------------
    2015-11-02 em       Initial implementation
    2015-11-20 rchallen Generalized for any given pixels

	"""
	# Set number of frames and image dimensions
	nframes, ny, nx = np.shape(data)

    # Set number of pixels
    npix = len(pixels)   

    # Initialize array for flux values of pixels
    p = np.zeros((nframes, npix))

    # Extract flux values from images
    for i in range(npix):
        for j in range(nframes):
            p[j,i] = data[j, pixels[i][0], pixels[i][1]]

    # Initialize array for normalized flux of pixels
    phat = np.zeros(p.shape)
    
	# Remove astrophysics by normalizing
	for t in range(nframes):
		phat[t]	= p[t]/np.sum(p[t])

    # Calculate mean flux through all frames at each pixel
	pbar	= np.mean(phat, axis=0)

    # Difference from the mean flux
	dP	= phat - pbar

	return(phat, dP) #decide which one it is? 

def eclipse(times, period, t0, i, r, p, phase_m, depth):
	
	"""
	times: array
		time when each frame was taken
	period: number
		period of the planet
	t0:	number
		time of transit (phase=zero)
	i:	number
		inclination
	r:	number
		a/Rstar
	p:	number
		Rplanet/Rstar
	phase_m:number
		phase of eclipse midpoint ~.5
	depth: number
		eclipse depth
	---------------
	http://iopscience.iop.org/article/10.1088/0004-637X/767/1/64/pdf;jsessionid=B1D9969B31D8FA4F21043BFC7EAF67F5.ip-10-40-2-121
	equations 3-5/6

	"""
	#secondary eclipse
	
    P_ecl	= np.zeros(np.size(times))

	phase	= ((times - t0) % period) / period

    r	= a * np.sqrt(1-(np.sin(i)**2) * (np.cos((phase-phase_m) * 2 * np.pi)**2))

    in_ecl	= np.where((r < 1-p) & (phase > .1) & (phase < .9))
    P_ecl[in_ecl] = 1

    #ingress and egress
    ing_egr	= np.where(  (r > 1-p) & (r < 1+p) & (phase>.1) & (phase < .9))

    t1=np.arccos((1+r[ing_egr]**2-p**2)/(2*r[ing_egr]))
    t2=np.arccos((r[ing_egr]**2+p**2-1)/(2*r[ing_egr]*p))
    P_ecl[ing_egr]=(1/(np.pi*p**2))*(t1-np.sin(t1)*np.cos(t1))+(1/np.pi)*(t2-np.sin(t2)*np.cos(t2))

    F_ecl=depth*(1-P_ecl)

	return(F_ecl)

# def zen(phat, pld_params, frame_times, eclipse_params, linramp, quadramp, offset):

# 	"""
# 	phat: 2d array
# 		normalized pixel values in each frame 
# 		or should we use pbar?
# 	pld_params: 1d array
# 		first derivative of signal wrt each pixel, fitted parameter	
# 	eclipse_params: 1D array
# 		go into eclipse/phase curve model to tell what eclipse is
# 	ramp: number?
# 		linear ramp term
# 	quadramp: number?
# 		quadratic ramp term
# 	offset: number?
# 		constant offset	
# 	"""
# 	N, times= np.shape(phat)
	
# 	dS	= np.zeros(times)

# 	eclipse = eclipse(frame_times,*eclipse_params)
	
# 	for t in range(times):
# 		for i in range(N):
# 			dS[t]	= dS[t] + pld_params[i] * phat[t]
		
# 		dS[t]	= dS[t] + eclipse[t] + linramp * t + quadramp * t**2 + offset


# 	return(dS)

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
    2015-11-02 em       Initial implementation.
    2015-11-20 rchallen Adaptation for use with MCcubed

    Notes:
    ------
    Only allows for quadratic ramp functions
    """

    # PLD term in the model
    PLD = 0

    # Calculate eclipse model using parameters
    # FINDME: remove hard-coded number of ramp params
    eclipsepars = par[npix+1:-3]
    
    eclipse = eclipse(x, *eclipsepars)
    
    # Calculate the sum of the flux from all considered pixels
    for i in range(npix):
        PLD += par[i]*phat[:,i]

    # Calculate the model
    # FINDME: allow for general ramp function
    y = PLD + eclipse + par[i+1]*x + par[i+2]*x**2 + par[i+3]

    return y









