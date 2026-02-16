import numpy as np
from math import pi, sqrt
import sys
import time

## ============ Generic Gaussian function
## ------ Inputs:
# x: argument
# mu: center of the Gaussian
# sig: standard deviation
def gaussian(x, mu, sig):
    pref=1/(sig*np.sqrt(2*pi))
    expo=np.exp(-((x-mu)**2)/(2*sig**2))
    return pref*expo

## ============ Statistical g-distribution due to temporal averaging
## ------ Inputs:
# g: a set of possible g-value spanning from 0 to g_center
# g_center: g-value one would get in absence of temporal averaging
# ratio_t: ratio between the light pulse duration (tau_p) and the electron pulse duration (tau_e) - Rt=tau_p/tau_e
def g_distribution_temporal_averaging(g,g_center,ratio_t):
    return ((1./sqrt(pi)*ratio_t)*(g/g_center)**(ratio_t*ratio_t))*1./(g*np.sqrt(np.log(g_center/g)))

## ============ Statistical g-distribution due to spatial averaging
## ------ Inputs:
# g: a set of possible g-value spanning from 0 to g_center
# g_center: g-value one would get in absence of spatial averaging
# ratio_s: ratio between the light beam waist (sigma_p) and the electron beam waist (sigma_e) - Rs=sigma_p/sigma_e
def g_distribution_spatial_averaging(g,g0,ratio_s):
    return (ratio_s*ratio_s)*(1./g)*((g/g0)**(ratio_s*ratio_s))

## ============ Draw a progress bar
## ------ Inputs:
# it: iteration
# prefix: title of the progressbar
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
        
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)