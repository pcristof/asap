from numba import jit
import numpy as np
from scipy import constants as cst

@jit(nopython=True, cache=True)
def integrate(x, u, v, dspeed=2000., auto=False):
    '''
    Function to integrate v on the x solution. Uses a numba accelerated loop.
    This method does not require evenly spaced inputs.
    Method computes the ± dspeed limits around x[i] and associate to x[i] the
    average corresponding values in v.

    Input parameters:
    - x     :   [1d arrray] wavelengths on which to integrate v.
    - u     :   [1d arrray] wavelengths solution associated to v
    - v     :   [1d arrray] fluxes associated to u.

    Output parameters:
    - y     :   [1d arrray] integrated fluxes associated to x.
    '''
    if x[0] < u[0]:
        print('integrate : a value in x is below integration range.')
    if x[-1] > u[-1]:
        print('integrate : a value in x is above integration range.')

    len_x = len(x)
    len_u = len(u)

    if auto:
        dspeed = np.median(np.diff(x)/x[:-1]*cst.c)

    y = np.zeros(np.shape(x)) # initialization
    ipbl = 0
    mastercount = 0
    countzero = 0
    for i in range(len_x):
        dxi = x[i] * dspeed / cst.c ## delta_lamda
        dxi = dxi/2 ## I consider the half speed
        sum_vals = 0.
        count = 0
        ## Handle 
        # if x[i] - dxi > u[ipbl]:
        #     mastercount += 1 
        #     j = ipbl
        # else:
        #     j = 0
        j = ipbl
        # j = 0
        while (u[j]<=x[i]+dxi) & (j<len_u):
            if (u[j]>=(x[i]-(dxi))) & (u[j]<=(x[i]+(dxi))):
                # check if value also in next iteration:
                if i<len_x:
                    dxi_next = (x[i+1] * dspeed / cst.c)/2
                    if u[j] < (x[i+1]-dxi_next):
                        ipbl = j
                sum_vals = sum_vals + v[j]
                count+=1
            j+=1
        # pos = ((u >= (x[i] - dxi)) & (u <= (x[i] + dxi)))
        # segment = np.copy(v[pos])
        if count==0:
            countzero+=1
            count = 1
            sum_vals = np.nan
        y[i] = sum_vals / count
    # if mastercount < len(x):
    #     print("integrate : could be optimized. ")
    #     print("Times speedup trick used : ")
    #     print(mastercount)
    #     print("Total number of calls : ")
    #     print(len(x))
    return y