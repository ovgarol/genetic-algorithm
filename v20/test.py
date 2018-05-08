# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 22:53:36 2018
Test file for genal v021

@author: ofgarcia
"""

###############################################################################
##      TEST FOR GENAL PACKAGE
###############################################################################

import genalpy as gn

## 3 dim optimization of schwefel funtion, teoric solution = (420.9687,...)
#comunity = gn.comManager(func=gn.schwefel,pop=100,cmin=[-500.,-500.,-500.],cmax=[500.,500.,500.],direction=-1)
#result = comunity.runSimple(T=100,maxFit=0)
#result = comunity.runAdaptative(T=100,maxFit=0,varLim=100,maxVar=150)

## 4 dim optimization of sphere funtion, teoric solution = (0,...)
#comunity = gn.comManager(func=gn.sphere,pop=100,cmin=[-5,-5,-5,-5],cmax=[5,5,5,5],direction=-1,mutVar=1.5)
#result = comunity.runSimple(T=50,maxFit=0)
#result = comunity.runAdaptative(T=100,maxFit=0,varLim=2.1,maxVar=3.51)
#result = comunity.runConstrained(gn.circular,radius=1.,T=100,maxFit=0)

## 2 dim optimization of not-centered-sphere funtion, teoric solution = center = (0.2,-1.0)
comunity = gn.comManager(func=gn.notCenteredSphere,center=[0.2,-1.],pop=100,cmin=[-1.,-1.],cmax=[1.,1.],direction=-1)
#result = comunity.runSimple(T=100,maxFit=0)
result = comunity.runAdaptative(T=100,maxFit=0,varLim=.10,maxVar=2.1)
#result = comunity.runConstrained(rest=gn.circular,radius=.95,T=100,maxFit=0,varLim=.10,maxVar=2.1)

#print [i.genes for i in comunity.individuals]
## Visualization examples
comunity.xprint()
#comunity.similarityMatrix(plot=True)
#comunity.plotFitnessLandscape2D()

## sensitivity analysis example
#a = gn.sensitivity(gn.schwefel,result,times=200)
#print a[1],a[2]
#plt.hist(a[0])

## testing for metaAnalysis, comparing adaptative vs. simple analysis for 
## optimization of matyas function minimum at (0,0)
#ll = gn.comManager(func=gn.matyas,pop=20,cmin=[-1.,-1.],cmax=[1.,1.],direction=-1)
#ll = gn.comManager(func=gn.sphere,pop=20,cmin=[-1,-1,-1,-1],cmax=[1,1,1,1],mutVar=0.05,mutrate=0.5,direction=-1)
#a, af, a_best = gn.runBlock(ll,mode="adaptative",times=100,T=100)
#b, bf, b_best = gn.runBlock(ll,mode="simple",times=100,T=100)
#print a_best, b_best

#import scipy.stats as stats
#z1 = stats.wilcoxon(a[0],b[0])
#z2 = stats.wilcoxon(a[1],b[1])
#z3 = stats.wilcoxon(af,bf)
#print z1, z2, z3

## plot correlogram
import seaborn as sns
from pandas import DataFrame
a = DataFrame(a)
sns.pairplot(a)
