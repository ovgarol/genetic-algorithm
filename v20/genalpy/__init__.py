# -*- coding: utf-8 -*-
"""
===============================================================================
Genetic algorithms for python 
===============================================================================
    Created by ovgarol under GNU GPL (2018).    
    Clean and simple implementation of genetic algorithms. 
    Includes test functions for optimization and some visualization routines. 
"""
###
"""
Created on Mon Jan 01 22:47:40 2018

Update version (v11) @ Jan 07 11:00:00
    Optimized importation
    Changes in the genetic operators names

Update version (v10) @ Jan 01 10:23:45 2018
    New test function schewefel https://www.sfu.ca/~ssurjano/schwef.html
    All module divided in 2 files genal and testfunc for better use.
    import using 
    >>> import v10 as gn
    on test file in the same folder than the one that contains the folder with
    the modules.
    run function of comManager class renamed as runSimple
 
Udate from genal (v02) and substitute it.

Original test version (v01) @ Dec 27 2017

New version (v02) @ Dec 29 11:05:09 2017
    self contained mutation rate and mutation variation
    works perfectly, some bugs corrected
    runAdaptative function added to comManager class
    GENDIM bug corrected!!!
    runMeta function created for runing algorithms many times
        returns average and std of fitness and best result
    Working fine!!!

@author: ofgarcia
"""
__all__=["genal","testfunc"]
from genal import *
from testfunc import *
