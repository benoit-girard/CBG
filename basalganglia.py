#!/usr/bin/python
# -*- coding: utf-8 -*-

# Rewriting in python of the basal ganglia CBG model (Girard et al., 2008)
# author : Benoît Girard <benoit.girard@isir.fr>
# 9th March 2009

import math
import numpy

#-------------------------------------------
class BasalGanglia:
  #-----------------------------------------
  def __init__(self,model,NbChannels,opt_params=[]):
    self.NbChannels = NbChannels # number of channels in competition

    self.model = model           # model type, can be :
                                 # * GPR : the (Prescott et al., 2006, Neural Netwk) model,
                                 # * CBG : the (Girard et al., 2008, Neural Netwk.) model,
                                 # * CBGcustom : a model with the CBG connections, using lPDS neurons, 
                                 #   whose precise parameters are specified by the opt_param list.

    self.paramInit(opt_params)   # parameter initialisation (connection weights, neuron biases)

    self.stateReset()            # reset to 0 of all the internal variables

    self.f=open('log/BG_'+model,'w')    # log file where the internal state will be stored if logAll function is used

  #-----------------------------------------
  def __del__(self):
    self.f.close()

  #-----------------------------------------
  def stateReset(self):
    self.FS = 0       # Fast Spiking striatal interneurons
    self.old_FS = 0   # Variables named "variable_old" are buffers used to store the previous output of the considered neurons

    self.D1 = numpy.zeros((self.NbChannels))  # medium spiny neurons of the striatum with D1 dopamine receptors
    self.D2 = numpy.zeros((self.NbChannels))  # medium spiny neurons of the striatum with D2 dopamine receptors
    self.STN = numpy.zeros((self.NbChannels)) # Sub-Thalamic Nucleus
    self.GPe = numpy.zeros((self.NbChannels)) # external Globus Pallidus 
    self.GPi = numpy.zeros((self.NbChannels)) # internal Globus Pallidus & reticular Substantia Nigra 

    self.old_D1 = numpy.zeros((self.NbChannels))
    self.old_D2 = numpy.zeros((self.NbChannels))
    self.old_STN = numpy.zeros((self.NbChannels))
    self.old_GPe = numpy.zeros((self.NbChannels))
    self.old_GPi = numpy.zeros((self.NbChannels)) 

  #-----------------------------------------
  def paramInit(self,opt_params):

    # invTau are 1/tau, tau being the neurons' time constants

    # W_A_B is the projection weight from neuron A to neuron B

    # I_A is the bias applied to neuron A

    if self.model == 'CBG':
      self.invTau = 1./0.020
      self.invTauSmall = 1./0.005

      self.DA = 0.2
      
      self.W_STN_GPe  =  0.7
      self.W_STN_GPi  =  0.7
      self.W_GPe_STN = 0.45
      self.W_GPe_D1 = 1. 
      self.W_GPe_D2 = 1. 
      self.W_GPe_FS = 0.05
      self.W_GPe_GPi = 0.08
      self.W_D1_GPe = 0.4
      self.W_D2_GPe = 0.4
      self.W_D1_GPi = 0.4
      self.W_D2_GPi = 0.
      self.W_FS_D1 = 0.5
      self.W_FS_D2 = self.W_FS_D1
      self.W_FC_STN = 0.58
      self.W_FC_D1 = 0.1
      self.W_FC_D2 = 0.1
      self.W_FC_FS = 0.01
      self.W_S_D1 = 0.9
      self.W_S_D2 = 0.9
      self.W_S_FS = 0.09

      self.I_D1  = -0.1
      self.I_D2  = -0.1
      self.I_STN =  0.5
      self.I_GPe =  0.1
      self.I_GPi =  0.1

    elif self.model == 'customCBG':
      if len(opt_params)<18:
        print 'customBG : parameter list absent or incomplete.'
        exit()

      self.invTau = 1./0.020
      self.invTauSmall = 1./0.005

      self.DA = 0.2
      
      self.W_S_D1    = opt_params[0]
      self.W_S_D2    = opt_params[0]
      self.W_FC_D1   = opt_params[1]
      self.W_FC_D2   = opt_params[1]
      self.W_S_FS    = opt_params[2] 
      self.W_FC_FS   = opt_params[3] 
      self.W_STN_GPe = opt_params[4]
      self.W_STN_GPi = opt_params[4]
      self.W_GPe_STN = opt_params[5]
      self.W_GPe_D1  = opt_params[6] 
      self.W_GPe_D2  = opt_params[7] 
      self.W_GPe_FS  = opt_params[8]
      self.W_GPe_GPi = opt_params[9]
      self.W_D1_GPe  = opt_params[10]
      self.W_D1_GPi  = opt_params[11]
      self.W_D2_GPe  = opt_params[12]
      self.W_FS_D1   = opt_params[13]
      self.W_FS_D2   = opt_params[13]
      self.W_FC_STN  = opt_params[14]

      self.I_D1  = -opt_params[15]
      self.I_D2  = -opt_params[15]
      self.I_STN =  opt_params[16]
      self.I_GPe =  opt_params[17]
      self.I_GPi =  opt_params[17]

    elif self.model =='GPR':
      self.invTau = 1./0.040
      self.DA = 0.2;
      
      self.W_STN_GPe  = 0.9
      self.W_STN_GPi  = 0.9
      self.W_GPe_STN = 1.
      self.W_GPe_D1 = 0
      self.W_GPe_D2 = 0
      self.W_GPe_FS = 0
      self.W_GPe_GPi = 0.3
      self.W_D1_GPe = 0.
      self.W_D2_GPe = 1.
      self.W_D1_GPi = 1.
      self.W_D2_GPi = 0.
      self.W_FS_D1 = 0
      self.W_FS_D2 = 0
      self.W_FC_STN = 0.5
      self.W_FC_D1 = 0.5
      self.W_FC_D2 = 0.5
      self.W_S_STN = 0.5
      self.W_S_D1 = 0.5
      self.W_S_D2 = 0.5

      self.I_D1  = -0.2
      self.I_D2  = -0.2
      self.I_STN =  0.25
      self.I_GPe =  0.2
      self.I_GPi =  0.2

    else:
      print self.model, ' type de modèle inconnu'
      exit()

  #-----------------------------------------
  # updates the model state, integrating over timestep "dt" and salience input "salience", 
  # using the (very) basic Euler method.
  # "FC_Input" : excitatory input from the frontal cortex
  # the update for the CBG and CBGcustom is based on lPDS neurons
  # the update for the GPR is based on leaky-integrator neurons

  def stepCompute(self,dt,saliences,FC_Input):

    #-----------------------------
    if (self.model == 'CBG') or (self.model == 'customCBG'):

      sumSTN = self.old_STN.sum()
      sumFS = self.W_FC_FS * FC_Input.sum() + self.W_S_FS * saliences.sum()
      sumGPe = self.old_GPe.sum()
      sumD1 = self.old_D1.sum()
      sumD2 = self.old_D2.sum()

      self.FS = min(max(self.FS + self.invTauSmall * (  sumFS 
                                                      - self.W_GPe_FS * sumGPe 
                                                      - self.FS
                                                     ) * dt,0),1)

      self.D1 = numpy.minimum(
                  numpy.maximum(self.D1 + self.invTau * (  (1+self.DA) * 
                                                   ( self.W_FC_D1 * FC_Input
                                                   + self.W_S_D1 * saliences
			  	                   - self.W_GPe_D1 * self.old_GPe 
                                                   )
                                                 - self.W_FS_D1 * self.old_FS 
                                                 - self.D1 + self.I_D1
                                                ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels))
      self.D2 = numpy.minimum(
                  numpy.maximum(self.D2 + self.invTau * ( (1-self.DA) *
                                                  ( self.W_FC_D2 * FC_Input
                                                   + self.W_S_D2 * saliences
                                                  - self.W_GPe_D2 * self.old_GPe 
                                                  )
                                                 - self.W_FS_D2 * self.old_FS 
                                                 - self.D2 + self.I_D2
                                                ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels))
      self.STN = numpy.minimum(
                  numpy.maximum(self.STN + self.invTauSmall * ( self.W_FC_STN * FC_Input
                                                        - self.W_GPe_STN * sumGPe
                                                        - self.STN + self.I_STN
                                                       ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels))
      self.GPe = numpy.minimum(
                  numpy.maximum(self.GPe + self.invTau * (  self.W_STN_GPe * sumSTN 
                                                   - self.W_D2_GPe * self.old_D2 
                                                   - self.W_D1_GPe * self.old_D1
				                   - self.GPe + self.I_GPe
                                                  ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels))
      self.GPi = numpy.minimum(
                  numpy.maximum(self.GPi + self.invTau * (  self.W_STN_GPi * sumSTN 
                                                   - self.W_GPe_GPi * sumGPe 
                                                   - self.W_D1_GPi * self.old_D1
                                                   - self.GPi + self.I_GPi
                                                  ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels))

      self.old_FS=self.FS
      self.old_D1 = numpy.copy(self.D1)
      self.old_D2 = numpy.copy(self.D2)
      self.old_STN = numpy.copy(self.STN)
      self.old_GPe = numpy.copy(self.GPe)
      self.old_GPi = numpy.copy(self.GPi)

    #-----------------------------
    elif self.model =='GPR':

      # Compuation of tau da/dt = I - a

      sumSTN = self.old_STN.sum()
      
      self.D1 = self.D1 + self.invTau * ( (1+self.DA) * (self.W_FC_D1 * FC_Input
                                                       + self.W_S_D1 * saliences)
                                          - self.D1
                                        ) * dt

      self.D2 = self.D2 + self.invTau * ( (1-self.DA) * (self.W_FC_D2 * FC_Input
                                                       + self.W_S_D2 * saliences)
                                          - self.D2
                                        ) * dt

      self.STN = self.STN + self.invTau * (   self.W_FC_STN * FC_Input
                                            + self.W_S_STN * saliences
                                            - self.W_GPe_STN * self.old_GPe
                                            - self.STN
                                          ) * dt

      self.GPe = self.GPe + self.invTau * (   self.W_STN_GPe * sumSTN
                                            - self.W_D2_GPe * self.old_D2
                                            - self.GPe
                                          ) * dt
      self.GPi = self.GPi + self.invTau * (   self.W_STN_GPi * sumSTN
                                            - self.W_GPe_GPi * self.old_GPe
                                            - self.W_D1_GPi * self.old_D1
                                            - self.GPi
                                          ) * dt
      # Computation of y=f(a)
      self.old_D1 = numpy.minimum( 
                        numpy.maximum( self.D1 + self.I_D1,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))
      self.old_D2 = numpy.minimum( 
                        numpy.maximum( self.D2 + self.I_D2,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))
      self.old_STN = numpy.minimum( 
                        numpy.maximum( self.STN + self.I_STN,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))
      self.old_GPe = numpy.minimum( 
                        numpy.maximum( self.GPe + self.I_GPe,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))
      self.old_GPi = numpy.minimum( 
                        numpy.maximum( self.GPi + self.I_GPi,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))

    else:
      print self.model, ' type de modèle inconnu'

  #-----------------------------------------
  def readGPi(self):
    return self.old_GPi
  #-----------------------------------------
  def readGPe(self):
    return self.old_GPe
  #-----------------------------------------
  def readSTN(self):
    return self.old_STN

  #-----------------------------------------
  # logs the internal state of the module
  # easily visualized with gnuplot : splot 'log/BG' matriw with lines
  def logAll(self):
    #if(timeStamp%10)==0:
    self.f.writelines(str(self.old_FS)+' ')
    self.f.writelines(' '.join([str(self.old_D1[i]) for i in range(self.NbChannels)])+' ')
    self.f.writelines(' '.join([str(self.old_D2[i]) for i in range(self.NbChannels)])+' ')
    self.f.writelines(' '.join([str(self.old_STN[i]) for i in range(self.NbChannels)])+' ')
    self.f.writelines(' '.join([str(self.old_GPe[i]) for i in range(self.NbChannels)])+' ')
    self.f.writelines(' '.join([str(self.old_GPi[i]) for i in range(self.NbChannels)])+'\n')

#---------------------------

def main():
  dt = 0.001
  BG = BasalGanglia('CBG',6)
  saliences = numpy.zeros((6))
  saliences[0]=0.4
  FC_Input = numpy.zeros((6))

  for t in range(100):
    BG.stepCompute(dt,saliences,FC_Input)
    BG.logAll()


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
