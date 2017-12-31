# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:41:44 2017
Original test version (v01) @ Dec 27 2017
New version (v02) @ Dec 29 11:05:09 2017
    self contained mutation rate and mutation variation
    works perfectly, some bugs corrected
    runAdaptative function added to comManager class
    GENDIM bug corrected!!!
    runBlock function created for running algorithms many times
        returns average, std of fitness and best result
    Working fine!!!
@author: ofgarcia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import array
import copy as copy

###############################################################################
##      global constants definition
###############################################################################    
        
GENDIM = 2 # chromosome dimension
MUTRATE = 0.10  # mutation rate
MUTA = .15 # maximum variability on mutation

###############################################################################
##      individual definition
###############################################################################    

class individual:
    def __init__(self, genes):
        self.genes = genes
        self.generation = 1
        self.fitness = 0.
        self.rank = 0
        self.flag = False
        
    def xprint(self):
        print self.rank , self.generation, "\t", self.fitness
        
    def delItem(self):
        #print self.genes
        del(self)

###############################################################################
##      community manager definition
###############################################################################    
        
class comManager:
    def __init__(self,pop=20,dim=GENDIM,mutrate=MUTRATE,mutVar=MUTA,cmin=False,cmax=False):
        self.population = pop
        self.dim = dim
        self.mutrate = mutrate
        self.mutVar = mutVar
        self.maxVar = 1.1
        if not cmin or not cmax:
            self.cmin = [0]*self.dim
            self.cmax = [1]*self.dim
        else:
            self.cmin = cmin
            self.cmax = cmax
            self.dim = len(cmin)
        self.runNumber = 0
        self.totalRuns = 0
        self.mode = "...NO RUN DETECTED..."
        self.elt = 1
        self.stats = []
        self.bestSol = []
        self.individuals = self.reset(self.population)
        self.currentBest = False
        self.histBest = []
        self.forMutate = int(self.mutrate*len(self.individuals))

    def reset(self,n):
        new_ind = [individual([0]*self.dim) for i in xrange(n)]
        for ind in new_ind:
            ind.genes = [np.random.uniform(self.cmin[i],self.cmax[i]) for i in xrange(self.dim)]        
        return new_ind
    
    def killInd(self,ind):
        self.individuals.remove(ind)
        ind.delItem()
    
    def getFitness(self,func,**kpars):
        for ind in self.individuals:
            if not ind.flag:
                ind.flag = True
                ind.fitness = func(ind.genes,**kpars)                

        self.individuals=sorted(self.individuals,key=lambda u: u.fitness)
        
        for i, ind in enumerate(self.individuals): ind.rank = i
        if len(self.individuals)>1:
            self.currentBest = copy.deepcopy(self.individuals[-1])
            self.histBest.append(self.currentBest)
        
    def getRestricted(self,rest,**kpars):
        new_ind = []
        for ind in self.individuals: 
            if rest(ind.genes,**kpars): new_ind.append(ind)
        
        self.individuals = new_ind
        
    def selection1(self):
        new_ind = []
        for ind in self.individuals: 
            if np.random.choice(self.population) < ind.rank: 
               new_ind.append(ind)
        self.individuals = new_ind
        
    def selection2(self):
        self.individuals=list(set(self.individuals))
        new_ind = []
        for ind in self.individuals: 
            if np.random.choice(self.population) < ind.rank: 
               new_ind.append(ind)
        self.individuals = new_ind
            
    def reproduce(self,sustPop=0.10):
        new_ind = []
        
        vacants = self.population-len(self.individuals)-self.elt
        if len(self.individuals)>int(sustPop*self.population+1):
            for i in xrange(vacants):
                mate1, mate2 = np.random.choice(self.individuals,2) 
                new_ind.append(crossover(mate1,mate2))
                #new_ind.append(averager(mate1,mate2)) # otro tipo de reproduccion           
        else:           
            new_ind = self.reset(vacants)
            
        self.individuals = new_ind+self.individuals

    def mutate(self):
        mut_ind = []
        for i in range(self.forMutate):
            mate1 = np.random.choice(self.individuals)
            mut_ind.append(mutate(mate1,self.mutVar))
            self.killInd(mate1)
            
        self.individuals = mut_ind+self.individuals
        if self.elt: self.individuals=self.individuals+[copy.deepcopy(self.currentBest)]
        
        for ind in self.individuals:
            if notInDomain(ind.genes,self.cmin,self.cmax):
                self.killInd(ind)
                
    def adaptate(self,var,varLim):
        if var < varLim and self.mutVar < self.maxVar:
            self.mutVar = self.mutVar*1.05
        
    def getStatistics(self):
        fl = []
        for ind in self.individuals: fl.append(ind.fitness)
        if len(fl)>1:
            var = np.std(fl)
            a = [np.median(fl), np.log10(var), min(fl), max(fl)]
            self.stats.append(a)
        return var
 
    def run(self,func,T=10,elite=1,maxFit=-1.0e-3,**kpars):
        self.mode = "Constrained"
        self.getFitness(func,**kpars)
        self.totalRuns = self.totalRuns+T
        self.elt = elite
        
        for i in range(T):
            self.selection1()
            self.reproduce()
            self.mutate()
            self.getFitness(func,**kpars)
            self.getStatistics()  
            self.runNumber = self.runNumber+1
            if self.currentBest.fitness >= maxFit: break

        self.bestSol = self.currentBest.genes
        return self.bestSol

    def runAdaptative(self,func,T=10,elite=1,maxFit=-1.0e-3,varLim=0.1,maxVar=1.1,**kpars):
        self.getFitness(func,**kpars)
        self.totalRuns = self.totalRuns+T
        self.elt = elite
        self.maxVar = maxVar
        self.mode = "Adaptative with varLim = "+str(varLim)+ " and maxVar = "+ str(maxVar)
        
        for i in range(T):
            self.selection1()
            self.reproduce()
            self.mutate()
            self.getFitness(func,**kpars)
            var = self.getStatistics()  
            self.runNumber = self.runNumber+1
            self.adaptate(var,varLim)
            if self.currentBest.fitness >= maxFit: break

        self.bestSol = self.currentBest.genes
        return self.bestSol
    
    def runConstrained(self,func,rest,T=5,elite=1,maxFit=-1.0e-2,**kpars):
        self.mode = "Constrained"
        self.getRestricted(rest,**kpars)
        self.getFitness(func,**kpars)
        self.totalRuns = self.totalRuns+T
        self.elt = elite
        
        for i in range(T):
            self.selection1()
            self.reproduce()
            self.mutate()
            self.getRestricted(rest,**kpars)
            self.getFitness(func,**kpars)
            self.getStatistics()
            self.runNumber = self.runNumber+1
            if self.currentBest.fitness >= maxFit: break
        
        self.bestSol = self.currentBest.genes
        return self.bestSol
                        
    def xprint(self):
        
        print ("\n***********************************************\n")
        print ("Exec abst...\n")    
        print "     Mode:", self.mode
        print "     Runs:", self.runNumber
        print "     Population size:", self.population
        print "\n... the top 5 ...\n"
        print "R  G\t Fitness"
        for i, ind in enumerate(self.individuals[-5:]): ind.xprint()
        
        print "\n     Best solution:"
        print self.bestSol
        self.showStats()

    def showStats(self):
        if len(self.stats):
            stats2 = array([[float(self.stats[j][i]) for j in xrange(len(self.stats))] for i in xrange(len(self.stats[0]))])
            plt.plot(stats2[3],"k-",label="max")
            plt.plot(stats2[0],"k--",label="med")
            #plt.plot(stats2[2],"k-",label="min",alpha=0.5)
            #plt.yscale('symlog')
            plt.legend(bbox_to_anchor=(1.1, 0.9),loc=2)
    
            ax2 = plt.twinx()
            ax2.plot(stats2[1],"b",label="std")
            ax2.set_ylabel(' ', color='b')
            ax2.tick_params('y', colors='b')
            ax2.legend(bbox_to_anchor=(1.1, 0.9),loc=3)
    
            plt.show()
        else:
            print "Insuficient data for statistical analysis..."
    
    def plotBest(self):
        gen = [[.0 for i in xrange(len(self.histBest))] for i in xrange(self.dim)]
        ger = [.0 for i in xrange(len(self.histBest))]
        
        for j,ind in enumerate(self.histBest):
            ger[j]= self.histBest[j].generation
            for i in range(self.dim):
                gen[i][j]=self.histBest[j].genes[i]
                
        surf=plt.matshow(gen,aspect=int(.5*len(self.histBest)/self.dim),cmap="hot")
        plt.colorbar(surf)
        
    def plotFitnessLandscape2D(self,func):

        MINX =  min([ind.genes[0] for ind in self.individuals])
        MAXX =  max([ind.genes[0] for ind in self.individuals])

        MINY =  min([ind.genes[1] for ind in self.individuals])
        MAXY =  max([ind.genes[1] for ind in self.individuals])

        deltax = (MAXX-MINX)/20.
        deltay = (MAXY-MINY)/20.
        x =  np.arange(MINX-deltax, MAXX+2*deltax, deltax)
        y =  np.arange(MINY-deltay, MAXY+2*deltay, deltay)

        X, Y = np.meshgrid(x,y)
        Z = func([X,Y])
        plt.contourf(X,Y,Z,20,cmap="viridis",aspect=1)
        for ind in self.individuals: plt.scatter(ind.genes[0],ind.genes[1],10,color='r')
        plt.show()
        
###############################################################################
##      Genetic operators definition
###############################################################################    
    
def mutate(mate1,mutVar):
    dim = len(mate1.genes)
    delta_genes = [mutVar*np.random.normal(0,1) for i in mate1.genes]
    new_genes = [mate1.genes[i]*(1+delta_genes[i]) for i in range(dim)]
    new_individual = individual(new_genes)
    new_individual.generation = mate1.generation
    return new_individual

def crossover(mate1,mate2):
    new_genes = [np.random.choice([x,y]) for x,y in zip(mate1.genes, mate2.genes)]
    new_individual = individual(new_genes)
    new_individual.generation = max([mate1.generation,mate2.generation])+1
    return new_individual
    
def averager(mate1,mate2):
    new_genes = [(x+y)/2 for x,y in zip(mate1.genes, mate2.genes)]
    new_individual = individual(new_genes)
    new_individual.generation = max([mate1.generation,mate2.generation])+1
    return new_individual

def notInDomain(genes,cmin,cmax):
    iterable = xrange(len(genes))
    for i in iterable:
        if genes[i] > cmax[i] or genes[i] < cmin[i]: 
            return True
    return False

def similarityMatrix(pop,plot=False):
    dimension = xrange(len(pop))
    sim = [[0. for i in dimension] for j in dimension]

    for i,x in enumerate(pop):
        for j,y in enumerate(pop): 
            X = array(x.genes)
            Y = array(y.genes)
            a = sum(X*Y)/np.sqrt(sum(X**2)*sum(Y**2))
            sim[i][j] = 1.-np.arccos(a)/np.pi
                       
    if plot:        
        surf = plt.matshow(sim,aspect=1,cmap="viridis")
        plt.clim(0,1)
        plt.colorbar(surf)

    return sim

###############################################################################
##     run many calculations sequentialy definition
############################################################################### 

def runBlock(comMan, FUNC, mode = "run", times = 30, T=20, maxFit=182.613, varLim=0.1):
    results = [ 0. for i in range(times)]
    fitness = [ 0. for i in range(times)]

    for i in range(times):
        ll = copy.deepcopy(comMan)
        if mode == "run": results[i]=ll.run(FUNC,T=T,maxFit=maxFit)
        elif mode == "adaptative": results[i]=ll.runAdaptative(FUNC,T=T,maxFit=maxFit,varLim=varLim)
        else: 
            print "No allowed mode..."
            return False    
        fitness[i] = ll.currentBest.fitness
        
    bestResult = results[fitness.index(max(fitness))]
   
    results = array(results).T
    fitness = array(fitness)
    
    print "Fitness stats:"
    print "\t", fitness.mean(), fitness.std()
    print "Results stats:"
    for i in results: 
        print "\t", i.mean(), i.std()
    
    
    return results, fitness, bestResult

###############################################################################
##     TestFunctions definition
############################################################################### 

def sphere(f,**kpars):
    a = f[0]**2+f[1]**2+f[2]**2+f[3]**2
    return -a

def beale(f,**kpars):
    a = (1.5-f[0]*(1-f[1]))**2+(2.25-f[0]*(1-f[1]**2))**2+(2.62-f[0]*(1-f[1]**3))**2
    return -a    

def matyas(f,**kpars):
    a = 0.26*(f[0]**2+f[1]**2)-0.48*f[0]*f[1]
    return -a    

def ackley(f,**kpars):
    a = -20*np.exp(-0.2*np.sqrt(0.5*(f[0]**2+f[1]**2)))-np.exp(0.5*(np.cos(2*np.pi*f[0])+np.cos(2*np.pi*f[1])))+20+np.e
    return -a 

def holder(f,**kpars):
    a = -np.abs(np.sin(f[0])*np.cos(f[1])*np.exp(np.abs(1-np.sqrt(f[0]**2+f[1]**2)/np.pi)))
    return -a 

def notCenterSphere(f,**kpars):
    a = (f[0]-2)**2+(f[1]+4.2)**2+(f[2]-3.3)**2+(f[3]+2.5)**2
    return -a

def rosenbrock(f,**kpars):
    a = (1-f[0])**2+100*(f[1]-f[0]**2)**2
    return -a

def limRosenbrock(f,**kpars):
    a = (f[0]-1)**3-f[1]+1 < 0 and f[0]+f[1]-2 < 0
    return a

def circular(f,radius=1,**kpars):
    a = f[0]**2+f[1]**2 < radius**2
    return a

def himmelblau(f,**kpars):
    a = (f[0]**2+f[1]-11)**2+(f[0]+f[1]**2-7)**2
    return a
    
###############################################################################
##      MAIN
############################################################################### 

if __name__ == "__main__":
    
    #GENDIM = 4
    # 4 dim optimization with spherical function, teoric solution = (0,0,0,0)
    #ll = comManager(pop=20,cmin=[-1,-1,-1,-1],cmax=[1,1,1,1],mutVar=0.05,mutrate=0.5)
    #ll.runAdaptative(sphere,T=100,maxFit=-1.0e-15,varLim=0.1,maxVar=1.)
    #ll.run(sphere,T=100,maxFit=-1.0e-10,varLim=0.1)
    #ll.runImproved(sphere,T=100)
    #ll.fitnessLandscape2D(sphere)
    #ll.runConstrained(sphere,circular,T=100,radius=1.)
    #similarityMatrix(ll.individuals,plot=True)

    #GENDIM = 2
    # 2 dim optimization of himmelblau funtion, teoric solution = (-0.270845,-0.923039)
    ll = comManager(pop=100,cmin=[-1.,-1.],cmax=[1.,1.])
    #ll.run(himmelblau,T=100,maxFit=181.613)
    ll.runAdaptative(himmelblau,T=100,maxFit=182.613,varLim=0.1,maxVar=2.)
    #ll.runConstrained(himmelblau,circular,radius=1.,T=100,maxFit=185)
    #ll.plotFitnessLandscape2D(himmelblau)
    #similarityMatrix(ll.individuals,plot=True)
    
    ll.xprint()   
    #ll.plotBest()
    plt.plot( [i.fitness for i in ll.histBest])
    
    """
    testing for metaAnalysis
    """
    """
    import scipy.stats as stats 

    #ll = comManager(pop=50,cmin=[-1.,-1.],cmax=[1.,1.])
    ll = comManager(pop=20,cmin=[-1,-1,-1,-1],cmax=[1,1,1,1],mutVar=0.05,mutrate=0.5)
    a, af, c = runBlock(ll,sphere,mode="adaptative",times=50,T=100)
    b, bf, c = runBlock(ll,sphere,mode="run",times=50,T=100)
    
    plt.plot(a[0],a[1],'r.',b[0],b[1],'b.')
    stats.spearmanr(a[0],a[1])
    stats.spearmanr(b[0],b[1])

    #plt.hist2d(a[0],a[1],20)
    #plt.hist2d(b[0],b[1],20)
    stats.wilcoxon(a[0],b[0])
    stats.wilcoxon(a[1],b[1])
    stats.wilcoxon(af,bf)
    """
