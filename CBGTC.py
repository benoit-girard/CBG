#!/usr/bin/python
# -*- coding: utf-8 -*-

# Simulation of a whole cortico-baso-thalamo-cortical loop (CBGTC)
# author : Benoît Girard <benoit.girard@isir.fr>
# 25th September 2009

import math
import random
import numpy
import numpy.random
import basalganglia
import thalamusFC

#-------------------------------------------
class CBGTC:
  #-----------------------------------------
  def __init__(self,model='CBG',NbChannels=6,opt_params=[]):
    self.model = model
    self.NbChannels = NbChannels
    # creates the BG and Th-FC modules :
    self.BG = basalganglia.BasalGanglia(model,NbChannels,opt_params[:18])
    self.THFC = thalamusFC.thalamusFC(model,NbChannels,opt_params[18:])
    # computes the inhibition at rest : any inhibition below this
    # level is considered as partial selection
    self.restInhibition = self.getInhibRest(0.001)
   
    print "============="
    print model+' model created'
    print self.NbChannels,'channels'
    print 'Inhibition at rest',self.restInhibition
    print "============="

  #-----------------------------------------
  # updates the model state, integrating over timestep "dt" and salience input "salience", 
  # using the (very) basic Euler method.

  def stepCompute(self,dt,saliences):
    inhibs = self.BG.readGPi()
    FCout = self.THFC.readFC()
    self.BG.stepCompute(dt,saliences,FCout)
    self.THFC.stepCompute(dt,saliences,inhibs)

  #---------------------------
  # simulates the CBGTC for a given number of steps (NbSteps)
  # logs the state of the model at each timestep if verbosity[0]=='v'
  # returns inhibition levels 

  def nbStepsCompute(self,dt,NbSteps,saliences,verbosity='stfu'):
 
    for t in range(NbSteps):
      self.stepCompute(dt,saliences)
      if verbosity[0] == 'v':
        self.logAll()
    return self.BG.readGPi()

  #---------------------------
  # simulates the CBGTC loop until convergence of all channels
  # i.e. until |GPi(t+dt)-GPi(t)| < threshold
  # stops before convergence if t>3s
  # logs the state of the model at each timestep if verbosity[0]=='v'
  # returns time to convergence and inhibition levels 
  def CvgCompute(self,dt,threshold,saliences,verbosity='stfu'):

    t = dt
    self.stepCompute(dt,saliences)
    if verbosity[0] == 'v':
        self.logAll()
    cvg = False

    while ((cvg == False) or (t<0.1)) and (t<3.0):
      t+=dt
      inhibs = self.BG.readGPi()

      self.stepCompute(dt,saliences)
      if verbosity[0] == 'v':
        self.logAll()

      new_inhibs = self.BG.readGPi()

      cvg = True
      for i in range(len(inhibs)) :
        if abs(inhibs[i]-new_inhibs[i]) >= threshold :
          cvg = False
          break
  
    #print t,new_inhibs
    return t,new_inhibs

  #---------------------------
  # returns the level of inhibition at rest in the GPi, GPe, STN
  def getInhibRest(self,dt):
    saliences = numpy.zeros((self.NbChannels))
    inh = self.nbStepsCompute(dt,1000,saliences,'stfu')
    return inh[0]

  #-----------------------------------------
  # logs the internal state of the loop
  # easily visualized with gnuplot : splot 'log/moduleName' matrix with lines
  def logAll(self):
    self.BG.logAll()
    self.THFC.logAll()

  #--------------------------------------------------------
  # simulates the selection test from the (Gurney et al, 2001b) paper
  # returns a score between 0 and 1, depending on the completion of the success criteria
  # verbosity 'v' logs internal state
  # verbosity 'vv' prints step results on the terminal
  def simpleTest(self,dt,verbosity='stfu'):
    score = 0
    
    # STEP 1
    #--------
    saliences = numpy.zeros((self.NbChannels))
    inhibs = self.nbStepsCompute(dt,2000,saliences,verbosity)
    if inhibs[0] > 0.01:
      score += 0.2
      if verbosity=='vv':
        print 'step 1 : inhibitory output at rest',inhibs[0]
    else :
      if verbosity=='vv':
        print 'step 1 : no inhibitory output at rest'      

    # STEP 2
    #--------
    saliences[0] = 0.4
    inhibs = self.nbStepsCompute(dt,2000,saliences,verbosity)
    if (inhibs[0] < self.restInhibition) and (inhibs[1] >= self.restInhibition):
      score += 0.2
      if verbosity=='vv':
        print 'step 2 : channel 1 selected'
    else :
      if verbosity=='vv':
        print 'step 2 : channel 1 not selected'

    # STEP 3
    #--------
    saliences[1] = 0.6
    inhibs = self.nbStepsCompute(dt,2000,saliences,verbosity)
    if (inhibs[0] > inhibs[1]) and  (inhibs[1] < self.restInhibition) :
      score+=0.1
      if inhibs[0] >= self.restInhibition:
        score+=0.1
        if verbosity=='vv':
          print 'step 3 : Channel 2 selected alone'
      else:
        if verbosity=='vv':
          print 'step 3 : Channel 2 more selected than channel 1'
    else:
      if verbosity=='vv':
        print 'step 3 : Channel 2 not selected, or channel 1 more selected than channel 2'
    

    # STEP 4
    #--------
    saliences[0] = 0.6
    inhibs = self.nbStepsCompute(dt,2000,saliences,verbosity)
    if (inhibs[0] < self.restInhibition) and (inhibs[1] < self.restInhibition):
      score+=0.1
      if (inhibs[0]-inhibs[1]<0.005):
        score+=0.1
        if verbosity=='vv':
          print 'step 4 : Channels 1 and 2 similarly selected'
      else:
        if verbosity=='vv':
          print 'step 4 : Channels 1 or 2 not similarly selected'
    else:
      if verbosity=='vv':
        print 'step 4 : Channels 1 or 2 not selected'

    # STEP 5
    #--------
    saliences[0] = 0.4
    inhibs = self.nbStepsCompute(dt,2000,saliences,verbosity)
    if (inhibs[0] > inhibs[1]) and (inhibs[1] < self.restInhibition) :
      score+=0.1
      if inhibs[0] >= self.restInhibition:
        score+=0.1
        if verbosity=='vv':
          print 'step 5 : Channel 2 selected alone'
      else:
        if verbosity=='vv':
          print 'step 5 : Channel 2 more selected than channel 1'
    else:
      if verbosity=='vv':
        print 'step 5 : Channel 2 not selected, or channel 1 more selected than channel 2'
      
    return score

  #-------------------------------------------------------
  # Computes the multiple successive vectors test 
  #
  # * score[0] evaluates the capacity of the system of selecting the
  # channel with the highest input
  # * score[1] evaluates the capacity of the system of separating the
  # channel with the highest input from the channel with the second
  # highest input
  # * score[2] evaluates the amplification of the salience signal in the
  # winning FC channel
  # * score[3] evaluates the contrast of amplification between highest and second highest input
  # * score[4] is the average time of convergence
  # * score[5] is an histogram of the time of convergence (values longer than 1s are grouped in the last bin)

  def TwoHundredMotelsTest(self,dt, steps, verbosity='stfu'):

    score = [0.,0.,0.,0.,0.,numpy.zeros((100))]
    numpy.random.seed(17) # you may change the seed at your convenience
    for i in range(steps):
      saliences = numpy.random.random_sample((self.NbChannels))
      tcvg, inhibs =  self.CvgCompute(dt,1e-5,saliences,'stfu')
      score[4] += tcvg
      score[5][min(int(tcvg*100.),99)]+=1

      #-----------------------------------------
      max1 = 0. # maximum salience
      max2 = 0. # second maximum
      i1 = []   # list of the indexes of the salience maximum in the salience vector
      i2 = []   # the same for the second maximum
      for j in range(len(saliences)):
        if saliences[j]>max1:
          max2 = max1
          i2 = i1
          max1 = saliences[j]
          i1 = [j]
        elif saliences[j] == max1:
          i1.append(j)
        elif saliences[j]>max2:
          max2 = saliences[j]
          i2 = [j]
        elif saliences[j] == max2:
          i2.append(j)

      if verbosity=='vv':
        print '---------------------------'
        print 'Step :',i
        print 'Saliences   :',saliences
        print 'Inhibitions :',inhibs
        print 'FC          :',self.THFC.readFC()
        print 'Amplification Contrast :', ((float(self.THFC.readFC()[i1[0]]-max1) / max1) - (float(self.THFC.readFC()[i2[0]]-max2) / max2))/ (float(self.THFC.readFC()[i1[0]]-max1) / max1)

      #-----------------------------------------
      if (saliences.max() < self.restInhibition) :
        score[0] += 1.
        score[1] += 1.
      else:
        for m1 in i1:
          if (inhibs[m1]<self.restInhibition) and (inhibs[saliences.argmin()]>inhibs[m1]):
            score[0] += 1. / len(i1)
          for m2 in i2:
            #print inhibs, min(max(0.,(inhibs[m2]-inhibs[m1])/(self.restInhibition-inhibs[m1])),1) / (len(i1)*len(i2))
            score[1] += min(max(0.,(inhibs[m2]-inhibs[m1])/(self.restInhibition-inhibs[m1])),1) / (len(i1)*len(i2))
            if (max1>0.) and (max2>0.) and (score[2]>0.) :
              score[3] += (   (float(self.THFC.readFC()[m1]-max1) / max1) 
                              - (float(self.THFC.readFC()[m2]-max2) / max2)
                              ) \
                              / (len(i1)*len(i2)) \
                              / (float(self.THFC.readFC()[m1]-max1) / max1)
            if max1>0. :
              score[2] += float(self.THFC.readFC()[m1]-max1) / max1 / len(i1)

    if verbosity[0]=='v':            
      print '=============================='
      print 'Selection of the max input:',score[0]/steps
      print 'Selection contrast:        ',score[1]/steps
      print 'Amplification of the max:  ',score[2]/steps
      print 'Amplification Contrast:    ',score[3]/steps
      print 'T cvg:                     ',score[4]*1000./steps
    return score[0]/steps, score[1]/steps, score[2]/steps, score[3]/steps, score[4]/steps, score[5]/steps

  #---------------------------
  # computes selection efficiency as in the test defined in (Prescott et al 2006 Neural Netw)

  def evaluate2ChannelsCompetition(self,dt):

    nbsteps = 21
    e1=numpy.zeros((nbsteps,nbsteps))
    e2=numpy.zeros((nbsteps,nbsteps))

    saliences = numpy.zeros((self.NbChannels))
    for c1 in range(0,nbsteps):
      print 'column',c1
      for c2 in range(0,nbsteps):
        saliences[0]= c1/float(nbsteps-1)
        saliences[1]= c2/float(nbsteps-1)
        tcvg, inhibs = self.CvgCompute(dt,1e-5,saliences,'stfu')
        #inhibs = self.nbStepsCompute(dt,2000,saliences,'stfu')
        e1[c1,c2] = min(1,max(1 - inhibs[0]/self.restInhibition,0))
        e2[c1,c2] = min(1,max(1 - inhibs[1]/self.restInhibition,0))
  
    f1 = open('log/e1_'+self.model,'w')
    f1.writelines(' '.join([str(e1[i,j]) for i in range(0,nbsteps)]) + '\n' for j in range(0,nbsteps))
    f1.close()
    
    f2 = open('log/e2_'+self.model,'w')
    f2.writelines(' '.join([str(e2[i,j]) for i in range(0,nbsteps)]) + '\n' for j in range(0,nbsteps))
    f2.close()

#=========================================
def main():
#=========================================

  dt = 0.001
  NbChannels = 6
  modeltype = 'CBG' # change this to GPR to simulate (Prescott et al, 2006) model

  if modeltype == 'CBG':
    myCBGTC = CBGTC(modeltype,NbChannels)
  else:
    myCBGTC = CBGTC('GPR')

  myCBGTC.simpleTest(dt,'vv')
  #myCBGTC.TwoHundredMotelsTest(dt,200,'v')
  #myCBGTC.evaluate2ChannelsCompetition(dt) # can be pretty long

  exit()

  #=========================================
  # CBGcustom models can be derived from the following original CBG parameters :
  #=========================================

  CBGparams = [
    0.9,  # S -> D1/D2 synaptic weight
    0.1,  # FC -> D1/D2 synaptic weight
    0.09, # S -> FS
    0.01, # FC -> FS
    0.7,  # STN -> GPE/GPi
    0.45, # GPe -> STN
    1.,   # GPe -> D1
    1.,   # GPe -> D2
    0.05, # GPe -> FS
    0.08, # GPe -> GPi
    0.4,  # D1 -> GPe
    0.4,  # D1 -> GPi
    0.4,  # D2 -> GPe
    0.5,  # FS -> D1/D2
    0.58, # FC -> STN
    0.1,  # -I_D1/D2
    0.5,  # I_STN
    0.1,  # I_GPe/GPi
    0.18, # BG-> Th
    0.6,  # FC -> Th
    0.35, # FC -> TRN
    0.6,  # Th -> FC
    0.35, # Th -> TRN
    0.35, # TRN -> Th
    0.1   # I_Th
    ] * numpy.ones([25]) 

  model = 'customCBG'
  customCBG = CBGTC(model,6,CBGparams)

  customCBG.simpleTest(dt,'v')
  exit()

#---------------------------

if __name__ == '__main__':
  # Import Psyco if available
  try:
    import psyco
    psyco.log()
    psyco.profile()
    psyco.full()
  except ImportError:
    print 'Psyco not available.'
  main()
