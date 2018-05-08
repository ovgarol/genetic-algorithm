# -*- coding: utf-8 -*-
"""
Principal module for genetic algorithm implementation.
"""
###
"""
Created on Tue Dec 19 10:41:44 2017

Update version (v11) @ Jan 07 11:00:00
    Optimized importation
    Optimized for using numpy efficiently
    Changes in the genetic operators names
    New function comManager.rouletteSelection
    New function comManager.setPars

Original test version (v01) @ Dec 27 2017

New version (v02) @ Dec 29 11:05:09 2017
    self contained mutation rate and mutation variation
    works perfectly, some bugs corrected
    runAdaptative function added to comManager class
    GENDIM bug corrected!!!
    runMeta function created for runing algorithms many times
        returns average and std of fitness and best result
    Working fine!!!

Update version (v10) @ Jan 01 10:23:45 2018

View __init__.py file for usage and other histlog

@author: ofgarcia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import array
import copy as copy

###############################################################################
##      global constants definition for default values
###############################################################################    

POP = 20 # pop size
GENDIM = 2 # chromosome dimension
MUTRATE = 0.1  # mutation rate
MUTVAR = .05 # initial variability on mutation
MAXVAR = 1.1 # maximum allowed variability on mutation
ELITE = 1 # Elitist selection
MAXFIT = 0.0 # maximum fitness
VARLIM = 0.1 # minimal allowed fitness variance
INCREMENTAL = 1.05 # factor of variation of muta
DIR = 1 # Direction of optimization
INF = -1. # default inferior limit
SUP = 1. # default superior limit
SUSTPOP = 0.1 # minimum population before reset
SELSIZE = 0.5 # population selected by roulette selection

###############################################################################
##      dummy function definition
############################################################################### 

def FUNC(f,**kpars):
    a = 0
    for i in f: a += i
    return a

def REST(f,**kpars):
    return True

###############################################################################
##      individual definition
###############################################################################    

class individual(object):
    """
    Candidate solution in a population. 
    """
    def __init__(self, genes):
        self.genes = genes
        self.generation = 1
        self.fitness = 0.
        self.rank = 0
        self.flag = False
        self.p = 0
        
    def xprint(self):
        print self.rank , self.generation, "\t", self.fitness
        
    def delItem(self):
        del(self)

###############################################################################
##      community manager definition
###############################################################################    

class comManager(object):
    """
    Manager for candidate solutions to solve for FUNC. Implements several routines of optimization
    based on genetic algorithms (runSimple, runConstrained and runAdaptative) and 
    routines for tracking and presenting results of the process (xprint, showStats,
    etc.).
    """
    def __init__(self,**kpars):
        self._initPars(**kpars)            
        self.runNumber = 0
        self.totalRuns = 0
        self.mode = "...NO RUN DETECTED..."
        self.stats = []
        self.bestSol = []
        self.individuals = self.reset(self.population)
        self.currentBest = False
        self.histBest = []
        self.forMutate = round(self.mutRate*len(self.individuals))
        
    def _initPars(self,**kpars):
        self.population = kpars.get('pop', POP)
        self.dim = kpars.get('dim', GENDIM)
        self.mutRate = kpars.get('mutRate', MUTRATE)
        self.mutVar = kpars.get('mutVar', MUTVAR)
        self.varLim = kpars.get('varLim', VARLIM)
        self.maxVar = kpars.get('maxVar', MAXVAR)
        self.incremental = kpars.get('incremental', INCREMENTAL)
        self.elt = kpars.get('elite', ELITE)
        self.direction = kpars.get('direction', DIR)
        self.maxFit = kpars.get('maxFit', MAXFIT)
        
        self._selSize = kpars.get('selSize', SELSIZE)
        self.selSize = int(self._selSize*self.population+0.5) 
        self._sustPop = kpars.get('sustPop', SUSTPOP)
        self.sustPop = int(self._sustPop*self.population+1)
        
        self._func = kpars.get('func', FUNC)
        self.fitFunc = lambda f: self.direction*self._func(f,**kpars)
        self._rest = kpars.get('rest', REST)
        self.restFunc = lambda f: self._rest(f,**kpars)
        
        self.cmin = kpars.get('cmin', None)
        self.cmax = kpars.get('cmax', None)
        self._INF = kpars.get('inf', INF)
        self._SUP = kpars.get('sup', SUP)
    
        if not self.cmin or not self.cmax:
            self.cmin = [self._INF]*self.dim
            self.cmax = [self._SUP]*self.dim
        else:
            self.dim = len(self.cmin)
            
    def setPars(self,**kpars):
        self.population = kpars.get('pop', self.population)
        self.dim = kpars.get('dim', self.dim)
        self.mutRate = kpars.get('mutRate', self.mutRate)
        self.mutVar = kpars.get('mutVar', self.mutVar)
        self.varLim = kpars.get('varLim', self.varLim)
        self.maxVar = kpars.get('maxVar', self.maxVar)
        self.incremental = kpars.get('incremental', self.incremental)
        self.elt = kpars.get('elite', self.elt)
        self.direction = kpars.get('direction', self.direction)
        self.maxFit = kpars.get('maxFit', self.maxFit)
        
        self._selSize = kpars.get('selSize', self._selSize)
        self.selSize = int(self._selSize*self.population+0.5)
        self._sustPop = kpars.get('sustPop', self._sustPop)
        self.sustPop = int(self._sustPop*self.population+1)
        
        self.cmin = kpars.get('cmin', self.cmin)
        self.cmax = kpars.get('cmax', self.cmax)

        if 'func' in kpars.keys():
            self._func = kpars.get('func')
            self.fitFunc = lambda f: self.direction*self._func(f,**kpars)
        if 'rest' in kpars.keys():
            self._rest = kpars.get('rest')
            self.restFunc = lambda f: self._rest(f,**kpars)
        
    def reset(self,n):
        new_ind = [individual([0]*self.dim) for i in xrange(n)]
        for ind in new_ind:
            ind.genes = [np.random.uniform(self.cmin[i],self.cmax[i]) for i in xrange(self.dim)]        
        return new_ind
    
    def killInd(self,ind):
        self.individuals.remove(ind)
        #ind.delItem()

    def getFitness(self):
        for ind in self.individuals:
            if not ind.flag:
                ind.flag = True
                ind.fitness = self.fitFunc(ind.genes)            

        self.individuals=sorted(self.individuals,key=lambda u: u.fitness)
        
        for i, ind in enumerate(self.individuals): ind.rank = i
        if len(self.individuals)>1:
            self.currentBest = copy.deepcopy(self.individuals[-1])
            self.histBest.append(self.currentBest)
        
    def getRestricted(self):
        """
        Check if an individual's gene is allowed.
        """
        new_ind = []
        for ind in self.individuals: 
            if self.restFunc(ind.genes): new_ind.append(ind)
        
        self.individuals = new_ind

    def rankSelection(self):
        """
        Genetic operator. Choses based in the rank the individuals that survive
        to the next generation.
        """
        new_ind = []
        for ind in self.individuals: 
            if np.random.choice(self.population) < ind.rank: 
               new_ind.append(ind)
        self.individuals = new_ind

    def rouletteSelection(self):
        """
        Genetic operator. Choses based in the roulette method the individuals that survive
        to the next generation.
        """
        new_ind = []
        minFitness = self.individuals[0].fitness
        prob = [ind.fitness-minFitness for ind in self.individuals]
        prob = array(prob)/sum(prob)
        size = self.selSize
        new_ind = np.random.choice(self.individuals,size=size,replace=False,p=prob)
        self.individuals = list(new_ind)
        
    def selection2(self):
        """
        Genetic operator. Choses based in the rank  the individuals that survive
        to the next generation. This method no return repeated individuals.
        """
        self.individuals=list(set(self.individuals))
        new_ind = []
        for ind in self.individuals: 
            if np.random.choice(self.population) < ind.rank: 
               new_ind.append(ind)
        self.individuals = new_ind
            
    def reproduce(self):
        """
        Genetic operator. Choses randomly some indiviudals and aply custom crossover operator.
        """
        new_ind = []
        lind = len(self.individuals)
        vacants = self.population-lind-self.elt
        if lind>self.sustPop:
            for i in xrange(vacants):
                mate1, mate2 = np.random.choice(self.individuals,2) 
                new_ind.append(crossover(mate1,mate2))
                #new_ind.append(averager(mate1,mate2)) # otro tipo de reproduccion           
        else:           
            new_ind = self.reset(vacants)
            
        self.individuals = new_ind+self.individuals

    def mutate(self):
        """
        Genetic operator. Choses randomly some indiviudals and aply custom mutation operator.
        """
        mut_ind = []
        for i in range(int(self.forMutate)):
            mate1 = np.random.choice(self.individuals)
            mut_ind.append(mutate(mate1,self.mutVar))
            self.killInd(mate1)
            
        self.individuals = mut_ind+self.individuals
        if self.elt: self.individuals=self.individuals+[copy.deepcopy(self.currentBest)]
        
        for ind in self.individuals:
            if notInDomain(ind.genes,self.cmin,self.cmax):
                self.killInd(ind)
                
    def adaptate(self,var):
        """
        Adjuste the mutation rate and variability in order to improve the algorithm.
        """
        if var < self.varLim:
            if self.mutVar < self.maxVar:
                self.mutVar *= self.incremental
            elif self.forMutate < 0.5*self.population:
                self.forMutate += 1
                        
    def getStatistics(self):
        """
        Calculate statistical parameters for fitness in a generation.
        """
        fl = []
        for ind in self.individuals: fl.append(ind.fitness)
        if len(fl)>1:
            var = np.std(fl)
            a = [np.median(fl), var, min(fl), max(fl)]
            self.stats.append(a)
            return var
        return 0
 
    def runSimple(self,T=10,**kpars):
        """
        Run a genetic algorithm for optimizing a function. Tipically, returns 
        least acurate results than the simple implementation (see 'runAdaptative' method).
        """       
        self.setPars(**kpars)
        self.mode = "Simple"
        self.getFitness()
        self.totalRuns = self.totalRuns+T
        
        for i in range(T):
            self.rankSelection()
            self.reproduce()
            self.mutate()
            self.getFitness()
            self.getStatistics()  
            self.runNumber += 1
            if self.currentBest.fitness >= self.maxFit: break

        self.bestSol = self.currentBest.genes
        return self.bestSol

    def runAdaptative(self,T=10,**kpars):
        """
        Run a genetic algorithm for optimizing a function auto ajusting the variability
        and the rate of the mutations in the population. Tipically, returns 
        most acurate results than the simple implementation (see 'runSimple' method).
        """
        self.setPars(**kpars)
        self.mode = "Adaptative with varLim = "+str(self.varLim)+ " and maxVar = "+ str(self.maxVar)
        self.getFitness()
        self.totalRuns = self.totalRuns+T
        
        for i in range(T):
            self.rankSelection()
            #self.rouletteSelection()
            self.reproduce()
            self.mutate()
            self.getFitness()
            var = self.getStatistics()  
            self.runNumber += 1
            self.adaptate(var)
            if self.currentBest.fitness >= self.maxFit: break

        self.bestSol = self.currentBest.genes
        self.mode = self.mode + "\n     Final: mutVar = "+str(self.mutVar)+" and forMutate = "+str(self.forMutate)
        return self.bestSol
    
    def runConstrained(self,T=5,**kpars):
        """
        Run a genetic algorithm for optimizing a function given some restrictions.
        """
        self.setPars(**kpars)
        self.mode = "Constrained"
        self.getRestricted()
        self.getFitness()
        self.totalRuns = self.totalRuns+T
        
        for i in range(T):
            #self.rankSelection()
            self.rouletteSelection()
            self.reproduce()
            self.mutate()
            self.getRestricted()
            self.getFitness()
            self.getStatistics()
            self.runNumber += 1
            if self.currentBest.fitness >= self.maxFit: break
        
        self.bestSol = self.currentBest.genes
        return self.bestSol
                        
    def xprint(self):
        """
        Print in terminal the results of a routine.
        """
        print ("\n***********************************************\n")
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
        """
        Plot a big picture of all the evolution.
        """
        if len(self.stats):
            stats2 = array(self.stats).T 
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
        
    def plotFitnessLandscape2D(self):
        """
        For 2D problems, plot the fitnes landscape as a contour plot. Shows each
        individual combination as a dot.
        """

        MINX =  min([ind.genes[0] for ind in self.individuals])
        MAXX =  max([ind.genes[0] for ind in self.individuals])

        MINY =  min([ind.genes[1] for ind in self.individuals])
        MAXY =  max([ind.genes[1] for ind in self.individuals])

        deltax = (MAXX-MINX)/20.
        deltay = (MAXY-MINY)/20.
        x =  np.arange(MINX-deltax, MAXX+2*deltax, deltax)
        y =  np.arange(MINY-deltay, MAXY+2*deltay, deltay)

        X, Y = np.meshgrid(x,y)
        Z = self.fitFunc([X,Y])
        plt.contourf(X,Y,Z,20,cmap="viridis")
        for ind in self.individuals: plt.scatter(ind.genes[0],ind.genes[1],10,color='r')
        plt.show()
        
    def similarityMatrix(self,plot=False):
        """ 
        Plot similarity matrix for a population.
        """
        pop = list(set(self.individuals))
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
##      Genetic operators definition
###############################################################################   

def mutate(mate1,mutVar):
    """
    Genetic operator. Returns an individual with slighly different genes 
    from progenitor. Chances exist of return a gene combination not contained 
    in the search space.
    """
    dim = len(mate1.genes)
    delta_genes = [mutVar*np.random.normal(0,1) for i in mate1.genes]
    new_genes = [mate1.genes[i]*(1+delta_genes[i]) for i in range(dim)]
    new_individual = individual(new_genes)
    new_individual.generation = mate1.generation
    return new_individual

def crossover(mate1,mate2):
    """
    Genetic operator. Returns an individual with some genes from each progenitor.
    Chances exist of return the exact same gene combination of one of the parents.
    """
    new_genes = [np.random.choice([x,y]) for x,y in zip(mate1.genes, mate2.genes)]
    new_individual = individual(new_genes)
    new_individual.generation = max([mate1.generation,mate2.generation])+1
    return new_individual
    
def averager(mate1,mate2):
    """
    Genetic operator. Returns an individual with the average genes value 
    from the progenitors.
    """
    new_genes = [(x+y)/2 for x,y in zip(mate1.genes, mate2.genes)]
    new_individual = individual(new_genes)
    new_individual.generation = max([mate1.generation,mate2.generation])+1
    return new_individual

def notInDomain(genes,cmin,cmax):
    """ 
    Check if a gene is not conteined in the search space. 
    """
    iterable = xrange(len(genes))
    for i in iterable:
        if genes[i] > cmax[i] or genes[i] < cmin[i]: 
            return True
    return False



###############################################################################
##     run many calculations sequentialy definition
############################################################################### 

def runBlock(comMan, mode = "simple", times = 30, Tot=20, **kargs):
    """
    Run many times the same routine. Used for meta analysis.
    """
    results = [0. for i in range(times)]
    fitness = [0. for i in range(times)]

    for i in range(times):
        ll = copy.deepcopy(comMan)
        if mode == "simple": results[i]=ll.runSimple(T=Tot)
        elif mode == "adaptative": results[i]=ll.runAdaptative(T=Tot)
        else: 
            print "No allowed mode..."
            return False, 0, 0
        fitness[i] = ll.currentBest.fitness
        
    bestResult = results[fitness.index(max(fitness))]
   
    results = array(results).T
    fitness = array(fitness)
    
    print "**********************************************************"
    print "Fitness stats:"
    print "\t", fitness.mean(), fitness.std()
    print "Results stats:"
    for i in results: print "\t", i.mean(), i.std()
    print "\n"
    
    return results, fitness, bestResult

###############################################################################
##     sensitivity analysis for a solution
############################################################################### 

def sensitivity(FUNC, genes, var = 0.1, times = 100,  **kargs):
    """
    Run a sensibility analysis for a solution.
    """
    A = []
    dim = len(genes)
    for i in range(times):
        delta_genes = [var*np.random.uniform(-1,1) for i in genes]
        new_genes = [genes[i]*(1+delta_genes[i]) for i in range(dim)]
        A.append(FUNC(new_genes,**kargs))
    
    return A, np.average(A), np.std(A)

