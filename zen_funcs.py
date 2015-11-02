#em 2 nov 2015

import numpy as np

def zen_init(values):
	
	"""
	values: 2d array
	"""
	
	N, frames= np.shape(values)
	
	#remove astrophysics by normalizing
	
	for t in range(frames):
		phat[:,t]	= values[:,t]/np.sum(values[:,t])

	pbar	= np.mean(phat, axis=1)
	
	dP	= phat - np.reshape(pbar,(N,1))

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

def zen(phat, pld_params, frame_times, eclipse_params, linramp, quadramp, offset):

	"""
	phat: 2d array
		normalized pixel values in each frame 
		or should we use pbar?
	pld_params: 1d array
		first derivative of signal wrt each pixel, fitted parameter	
	eclipse_params: 1D array
		go into eclipse/phase curve model to tell what eclipse is
	ramp: number?
		linear ramp term
	quadramp: number?
		quadratic ramp term
	offset: number?
		constant offset	
	"""
	N, times= np.shape(phat)
	
	dS	= np.zeros(times)

	eclipse = eclipse(frame_times,*eclipse_params)
	
	for t in range(times):
		for i in range(N):
			dS[t]	= dS[t] + pld_params[i] * phat[t]
		
		dS[t]	= dS[t] + eclipse[t] + linramp * t + quadramp * t**2 + offset


	return(dS)









