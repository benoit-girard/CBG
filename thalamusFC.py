#!/usr/bin/python
# -*- coding: utf-8 -*-

# Rewriting in python of the basal ganglia CBG model (Girard et al., 2008, Neural Netwk.)
# This is the thalamus-frontal cortex module
# author : Benoît Girard <benoit.girard@isir.fr>
# 9th March 2009

import math
import numpy

#-------------------------------------------
class thalamusFC:
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

    self.f=open('log/ThFC_'+model,'w')  # log file where the internal state will be stored if logAll function is used

  #-----------------------------------------
  def __del__(self):
    self.f.close()

  #-----------------------------------------
  def stateReset(self):
    self.Th = numpy.zeros((self.NbChannels)) # Neurons of the thalamic nucleus implied in the considered loop
    self.FC = numpy.zeros((self.NbChannels)) # Neurons of the frontal cortex area implied in the considered loop

    # Variables named "variable_old" are buffers used to store the previous output of the considered neurons
    self.old_Th = numpy.zeros((self.NbChannels)) 
    self.old_FC = numpy.zeros((self.NbChannels))

    # TRN: Thalmic Reticular Nucleus
    # The TRN is made of NbChannels neurons in the GPR, 
    # while it is one neuron only in the CBG:
    if (self.model == 'CBG') or (self.model == 'customCBG'):
      self.old_TRN = 0
      self.TRN = 0
    elif self.model =='GPR':
      self.TRN = numpy.zeros((self.NbChannels))
      self.old_TRN = numpy.zeros((self.NbChannels))      
    else:
      print 'stateReset(): ', self.model, ' model type unknown'


  #-----------------------------------------
  def paramInit(self,opt_params):

    # invTau are 1/tau, tau being the neurons' time constants

    # W_A_B is the projection weight from neuron A to neuron B

    # I_A is the bias applied to neuron A

    if self.model == 'CBG':
      self.invTau = 1./0.080
      self.invTauSmall = 1./0.005

      self.W_BG_Th = 0.18
      self.W_BG_TRN = 0

      self.W_FC_Th  = 0.6
      self.W_FC_TRN = 0.35
  
      self.W_Th_FC  = 0.6
      self.W_Th_TRN  = 0.35

      self.W_TRN_Th = 0.35
      self.W_TRN_Th_self = 0

      self.I_Th = 0.1

    elif self.model == 'customCBG':
      if len(opt_params)<7:
        print 'customBG : parameter list absent or incomplete.'
        exit()

      self.invTau = 1./0.080 
      self.invTauSmall = 1./0.005

      self.W_BG_Th = opt_params[0]

      self.W_FC_Th  = opt_params[1]
      self.W_FC_TRN = opt_params[2]
  
      self.W_Th_FC  = opt_params[3]
      self.W_Th_TRN  = opt_params[4]

      self.W_TRN_Th = opt_params[5]

      self.I_Th = opt_params[6]
        
    elif self.model =='GPR':
      self.invTau = 1./0.040

      self.W_BG_Th = 1
      self.W_BG_TRN= 0.2

      self.W_FC_Th  = 1
      self.W_FC_TRN = 1
  
      self.W_Th_FC  = 1
      self.W_Th_TRN  = 1

      self.W_TRN_Th  = 0.4
      self.W_TRN_Th_self  = 0.125

      self.I_Th  = 0.
      self.I_FC  = 0.
      self.I_TRN = 0.

    else:
      print 'paramInit(): ', self.model, ' model type unknown'
      exit()

  #-----------------------------------------
  # updates the model state, integrating over timestep "dt" and salience input "salience", 
  # using the (very) basic Euler method.
  # "BG_Input" : inhibitory input from the BG (from the GPi/SNr)
  # the update for the CBG and CBGcustom is based on lPDS neurons
  # the update for the GPR is based on leaky-integrator neurons

  def stepCompute(self,dt,saliences,BG_Input):

    #-----------------------------
    if (self.model == 'CBG') or (self.model == 'customCBG'):
      saturation = 2 # set it to 1 for the original model, use 2 to have sound contrast amplification scores
      self.TRN = min(max(self.TRN + self.invTauSmall * ( self.W_FC_TRN * self.old_FC.sum() 
                                                       + self.W_Th_TRN * self.old_Th.sum()
                                                       - self.TRN
                                                       ) * dt,0),saturation)
      
      self.Th = numpy.minimum(
                  numpy.maximum(self.Th + self.invTauSmall * ( self.W_FC_Th * self.old_FC
                                                             - self.W_TRN_Th * self.old_TRN
                                                             - self.W_BG_Th * BG_Input
                                                             - self.Th + self.I_Th
                                                             ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels)*saturation)

      self.FC = numpy.minimum(
                  numpy.maximum(self.FC + self.invTau * ( self.W_Th_FC * self.old_Th
                                                        + saliences
                                                        - self.FC
                                                        ) * dt,
                                numpy.zeros(self.NbChannels)),
                  numpy.ones(self.NbChannels)*saturation)

      self.old_TRN = self.TRN
      self.old_Th  = numpy.copy(self.Th)
      self.old_FC  = numpy.copy(self.FC)

    #-----------------------------
    elif self.model =='GPR':

      # Computation of tau da/dt = I - a
      sumTRN = self.old_TRN.sum()

      self.Th = self.Th + self.invTau * (  self.W_FC_Th * self.old_FC
                                         - self.W_TRN_Th * sumTRN
                                         - (self.W_TRN_Th_self - self.W_TRN_Th) * self.old_TRN
                                         - self.W_BG_Th * BG_Input
                                         - self.Th
                                        ) * dt
      self.FC = self.FC + self.invTau * (  self.W_Th_FC * self.old_Th
                                         + saliences
                                         - self.FC
                                        ) * dt
      self.TRN = self.TRN + self.invTau * (  self.W_FC_TRN * self.old_FC
                                           + self.W_Th_TRN * self.old_Th
                                           - self.W_BG_TRN * BG_Input
                                           - self.TRN
                                          ) * dt
      # Computation of y=f(a)
      self.old_Th = numpy.minimum( 
                        numpy.maximum( self.Th + numpy.ones(self.NbChannels) * self.I_Th,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))
      self.old_FC = numpy.minimum( 
                        numpy.maximum( self.FC + numpy.ones(self.NbChannels) * self.I_FC,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))
      self.old_TRN = numpy.minimum( 
                        numpy.maximum( self.TRN + numpy.ones(self.NbChannels) * self.I_TRN,
                                       numpy.zeros(self.NbChannels)
                                     ),
                        numpy.ones(self.NbChannels))


    #-----------------------------
    else:
      print 'paramInit(): ', self.model, ' model type unknown'

  #-----------------------------------------
  def readFC(self):
    return self.FC

  #-----------------------------------------
  def logAll(self):
  # logs the internal state of the module
  # easily visualized with gnuplot : splot 'log/BG' matriw with lines
    self.f.writelines(str(self.old_TRN)+' ')
    self.f.writelines(' '.join([str(self.old_Th[i]) for i in range(self.NbChannels)])+' ')
    self.f.writelines(' '.join([str(self.old_FC[i]) for i in range(self.NbChannels)])+'\n')

#---------------------------

def main():
  dt = 0.001
  THFC = thalamusFC('CBG',6)
  saliences = numpy.zeros((6))
  saliences[0] = 0.4
  BG_Input = numpy.zeros((6))

  for t in range(200):
    THFC.stepCompute(dt,saliences,BG_Input)
    THFC.logAll()

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
