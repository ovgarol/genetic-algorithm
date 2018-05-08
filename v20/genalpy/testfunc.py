# -*- coding: utf-8 -*-
"""

"""
###
"""
Update for v10 on Jan 04 11:30:49
    Documentation added

Created on Mon Jan 01 22:49:43 2018
New for v10 contains test functions for optimization.
Update version (v02.1) @ Jan 01 10:23:45 2018
    New test function schewefel https://www.sfu.ca/~ssurjano/schwef.html
    All module divided in 2 files genal and testfunc for better use.
    import using 
    >>> import v10 as gn
    on test file in the same folder than the one that contains the folder with
    the modules.

@author: ofgarcia
"""

import numpy as np
import time as time

def timeit(some_function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        RESULT = some_function(*args, **kwargs)
        t2 = time.time()
        print("Time it took to run the function: " + str((t2 - t1)))
        return RESULT 
    return wrapper
    
###############################################################################
##     TestFunctions definition
############################################################################### 

def sphere(f,**kpars):
    """ 
    Sphere function: n-dimentional, convex, continuos, differentiable, separable, unimodal.
    Global minimum at (0,...,0).    
    http://www.sfu.ca/~ssurjano/spheref.html
    """
    a = 0
    for i in f: a+= i**2
    return a

def beale(f,**kpars):
    """ 
    Beale function: 2-dimentional, not-convex, continuos, differentiable, separable, unimodal.
    http://www.sfu.ca/~ssurjano/ackley.html    
    """
    a = (1.5-f[0]*(1-f[1]))**2+(2.25-f[0]*(1-f[1]**2))**2+(2.62-f[0]*(1-f[1]**3))**2
    return a    

def matyas(f,**kpars):
    """ 
    Matyas function: 2-dimentional, continuous, differentiable, non-separable, unimodal, convex.
    Global minimum at (0,0).
    http://www.sfu.ca/~ssurjano/matya.html
    """          
    a = 0.26*(f[0]**2+f[1]**2)-0.48*f[0]*f[1]
    return a    

def ackley(f,**kpars):
    """
    Ackley function: 2-dimentional
    """
    a = -20*np.exp(-0.2*np.sqrt(0.5*(f[0]**2+f[1]**2)))-np.exp(0.5*(np.cos(2*np.pi*f[0])+np.cos(2*np.pi*f[1])))+20+np.e
    return -a 

def holder(f,**kpars):
    a = -np.abs(np.sin(f[0])*np.cos(f[1])*np.exp(np.abs(1-np.sqrt(f[0]**2+f[1]**2)/np.pi)))
    return a 

def notCenteredSphere(f, center, **kpars):
    a = 0
    for i, j in zip(f,center):
        a += (i-j)**2
    return a

def rosenbrock(f,**kpars):
    a = (1-f[0])**2+100*(f[1]-f[0]**2)**2
    return a

def limRosenbrock(f,**kpars):
    a = (f[0]-1)**3-f[1]+1 < 0 and f[0]+f[1]-2 < 0
    return a

def circular(f,radius=1,**kpars):
    a = 0
    for i in f:
        a += i**2 
    x = a < radius**2
    return x

def himmelblau(f,**kpars):
    a = (f[0]**2+f[1]-11)**2+(f[0]+f[1]**2-7)**2
    return a

def schwefel(f,**kpars):
    d = len(f)
    A = 0.
    for i in f: A += i*np.sin(np.sqrt(abs(i)))
    a = d*418.9829-A
    return a
        