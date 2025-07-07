from numba import jit
import numpy as np
from scipy.optimize import curve_fit

@jit(nopython=True)
def normalize_axis(a,b):
    '''
    method that modify array based on second array
    returns (a-np.mean(b))/(np.max(a)-np.min(b))    
    '''
    c = (a-np.mean(b))/(np.max(b)-np.min(b))
    return c

@jit(nopython=True)
def revert_normalize_axis(a,b):
    '''
    method that modify array based on second array
    returns a*(np.max(b)-np.min(b))+np.mean(b)   
    '''
    c = a*(np.max(b)-np.min(b))+np.mean(b)
    return c

@jit(nopython=True, cache=True)
def fit_1d_polynomial(x, y, degree=3, returnCovMat=False, normalize_axes=False):
    '''
    Input parameters:
    - x         :   Must be a 1D array
    - z         :   Must be a 1D array
    - degree    :   degree used to fit polynomial
    Output parameters:
    - o  :  1D array containing the values of the ten coefficients 
            a,d,f,b,c,e,g,h,i,j
    '''
    if normalize_axes:
        x = normalize_axis(x, x)
    
    ## We are solving Y=XA
    ## A are the coefficients
    ## Y are the values associated to X
    ## X shape is going to be (len(Y), degree)
    # X = np.empty((len(x), degree+1))
    # for i in range(degree+1):
    #     X[:,i] = x**i
    ## Apparently an alternative is
    X = np.vander(x, degree+1)
    ## Y is simply Y
    Y = y
    ## And the solution is simply
    ## A=(X.T*X)^{−1}X.T*Y
    X_T = X.T
    X_T_X = np.dot(X.T, X)
    X_T_X_inv = np.linalg.inv(X_T_X)
    X_T_Y = np.dot(X_T, Y)
    # X_T_X_inv_X_T = np.dot(X_T_X_inv, X_T)
    A = np.dot(X_T_X_inv, X_T_Y)

    return A

@jit(nopython=True, cache=True)
def poly1d(x, coeffs, normalize_axes=False, normalize_range=None):
    if normalize_axes:
        if normalize_range is None:
            x = normalize_axis(x, x)
        else:
            x = normalize_axis(x, normalize_range)
    degree = len(coeffs)-1
    output = np.zeros(len(x))
    for i in range(degree+1):
        output+=coeffs[i]*x**(degree-i)
    return output



def fit_poly(x, z, degree=3):
    '''
    Returns the coefficients of the best fit obtained with 
    scipy.optimize.curve_fit. 

    Input parameters:
    - x         :   x values for data to fit
    - y         :   y values for data to fit
    - degree    :   degree of the polynomial used for fit

    Output parameters:
    - popt      :   optimal coefficients of the polynomial. The length of 
                    popt depends on the degree of the polynomial used.
    - pcov      :   covariance matrix on the parameters. The size of pcov
                    depends on the degree of the polynomial used. 
    '''
    ## Define local function based on the polynomial degree 
    if degree==6: 
        def _poly(x,k,a,b,c,d,e,f): return k*x**6+a*x**5+b*x**4+c*x**3+d*x**2+e*x+f
    elif degree==5: 
        def _poly(x,a,b,c,d,e,f): return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f
    elif degree==4: 
        def _poly(x,b,c,d,e,f): return b*x**4+c*x**3+d*x**2+e*x+f
    elif degree==3: 
        def _poly(x,c,d,e,f): return c*x**3+d*x**2+e*x+f
    elif degree==2: 
        def _poly(x,d,e,f): return d*x**2+e*x+f
    elif degree==1: 
        def _poly(x,e,f): return e*x+f
    elif degree==0: 
        def _poly(x,f): return f
    else : raise Exception("fit_poly --> Polynomial degree not supported")

    ## Fit function on data    
    popt, pcov = curve_fit(_poly, x, z) 
    return popt, pcov
