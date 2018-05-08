# -*- coding: utf-8 -*-
"""
Created on  Mon Jan 01 22:23:32 2018
Test file for genal v021

@author: ofgarcia
"""

import genalpy as gn
import numpy as np
import matplotlib.pyplot as plt
from scipy import array
from pandas import read_csv
from numba import jit

FILE = r'C:\Users\ofgarcia\Desktop\myHack\dga\v11\datos_perfil.csv'
df = read_csv(FILE)
selected = (df["month"]=="FEB")
X = df.loc[selected]

depth = X["depth"]
par = X["par"]
chla = X["chla"]

depth = array(depth)
par = array(par)
chla = array(chla)
Schla = sum(chla)

@jit(nopython=True)
def model(f,depth=depth):
    #PAR = f[3]*np.exp(-depth/f[5])
    CHLA = f[0]*np.exp(depth*(f[1]-f[2]))*(f[3]+f[4]*np.exp(depth/f[5]))**(-f[1]*f[5])
    return CHLA

def fitness(f,**kargs):
    #Spar = np.sum(par)*max(par)
    CHLA = model(f,depth=depth)
    #a = np.sum((PAR-par)**2)/Spar  
    b = sum(abs(CHLA-chla))/Schla
    return b

def fitnessL(f,**kpars):
    CHLA = model(f,depth=depth)
    a = np.sum([(CHLA[i]-chla[i])**2 for i in [0,5,10,20,30,50,100]])
    return a

# C,r,m,Io,k,S
f=[250,0.4/3,0.2/3,4100.,50.,6.7]

ll = gn.comManager(func=fitness,pop=250, maxFit = 0,
                   cmin=[0.,0.,0.,0.,0.,1.],
                   cmax=[1000.,1.5,1.0,8000.,100.,10.],
                   direction=-1,
                   )
result = ll.runSimple(T=150,maxFit=0) 
result = ll.runAdaptative(T=150) 
#a, af, a_best = gn.runBlock(ll,mode="adaptative",times=100,T=500)

ll.xprint()



#fit = lambda x: np.log(fitness(x))
#a = gn.sensitivity(fitness,result,times=300)
#plt.hist(a[0])
#print a[1], a[2]
#ll.similarityMatrix(ll.individuals,plot=True)


t = model(result,depth)
plt.plot(depth,t,'r',depth,chla,'k.')
#plt.plot(depth,np.log(t[0]),'r',depth,np.log(par),'k')
#plt.plot(depth,np.log10(t[0]),'r',depth,np.log(par),'k')
#S=ll.similarityMatrix(plot=True)

"""
a, af, result = gn.runBlock(ll,mode="adaptative",times=100,T=500)

import seaborn as sns
from pandas import DataFrame
a = DataFrame(a).T
sns.pairplot(a)
"""

