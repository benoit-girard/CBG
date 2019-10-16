This code is a Python re-implementation of the contracting basal ganglia (CBG) model presentend in (Girard et al., 2008).

# First contact:

running "python CBGTC.py" should simulate the contracting basal
ganglia model (CBG) in three tests: 

* The first one replicates the test used in (Gurney et al 2001b) to
  generate their fig. 2a, also used in (Girard et al 2008) to generate
  fig 3. The terminal output should provide information about the
  performance of the program in each of the five steps. To generate
  new versions of the aforementioned figures, log files have to be
  used...

* The second one successively provides the model with 200 random input
  vectors, and computes 5 different scores: 

    - the first one reflects how frequently the maximum input is the
      disinhibited one after convergence,

    - the second one computes the selection contrast between the
      channel with maximum input and its strongest competitor,

    - the third one computes the amplification in the frontal cortex
      of the channel with maximum input with regard to its raw input,

    - the fourth one compares the amplification of the winning channel
      with its strongest competitor's amplification (amplification
      contrast),

    - the last one is the average time needed to reach the convergence
      of selection process.

  This test writes on the terminal the average value of each of these
  scores.

* The third one replicates (Prescott et al 2006) systematic salience
  search used to generate their figure 7, as well as (Girard et al
  2008) fig. 4. Only efficiency of channels 1 and 2 is computed with
  the provided code, using them to compute efficiency and distortion
  is straightforward (eqns. 7 and 8 in the Prescott et al paper).

The corresponding log files are stored in the log/ directory:

* BG and ThFC respectively contain the activity of the each neuron of
  the basal ganglia and thalamo-cortical modules at each time
  step. The GPi/SNr output used to generate (Girard et al 2008) fig. 3
  has to be read in columns 26, 27 and 28 of BG_CBG.

  With gnuplot, this should do the trick : 
  plot 'BG_GPR' using 26 with lines, '' u 27 w l, '' u 28 w l

* e1 and e2 contain the matrices of the efficiency scores of channel 1
  and 2 for the third test.

The program will try to use the psycho library to increase computation
speed, if not available, a 'Psyco not available.' message will be
displayed, but the program should compute without problem.

# Some more details:

The basalganglia.py and thalamusFC.py respectively define the
basalganglia and thalamusFC classes. The basalganglia corresponds to
the intrinsic basal ganglia circuitry (Striatum, GPe, GPi, SNr), the
thalamusFC to the thalamo-cortical loop (Th, TRN and FC).

The CBGTC.py defines the CBGTC class, which contains one instance of
the basalganglia object and one instance of the thalamusFC, allowing
the simulation of a complete cortico-baso-thalamo-cortical loop.  It
contains the various test functions allowing the replication of some
of the papers figures.

The program contains the code to also run the GPR model (Gurney et al
2001a), just create a CBGTC object with the model set to 'GPR' (just
look at the main() code in CBGTC.py).

You can also use some custom CBG model, i.e. using lPDS neurons and
the CBG structure, but with your own parameters. Create a CBGTC object
with model set to 'CBGcustom', and with a vector containing the 25
parameters (once again, look at the main() in CBGTC.py).

These three classes (basalganglia, thalamusFC, CBGTC) have the same
generic architecture:

stateReset() sets all neurons to 0

paramInit(opt_param) defines the synaptic weights and the neuron
biases, opt_parameter being used for the CBGcustom type of model only.

stepCompute(dt,inputs) updates the model state, integrating over
timestep "dt" and salience input "salience", using the (very) basic
Euler method.

logAll() obviously logs the internal state of the model in the
dedicated log files in the log/ directory.

The programs are commented (at least a bit), do not hesitate to dig
into them to get more details about the other functions.

# References

Girard, B., Tabareau, N., Pham, Q. C., Berthoz, A., & Slotine, J. J. (2008). Where neuroscience and dynamic system theory meet autonomous robotics: a contracting basal ganglia model for action selection. *Neural Networks*, 21(4), 628-641.

Gurney, K., Prescott, T. J., & Redgrave, P. (2001a). A computational model of action selection in the basal ganglia. I. A new functional anatomy. Biological cybernetics, 84(6), 401-410.

Gurney, K., Prescott, T. J., & Redgrave, P. (2001b). A computational model of action selection in the basal ganglia. II. Analysis and simulation of behaviour. Biological cybernetics, 84(6), 411-423.

Prescott, T. J., González, F. M. M., Gurney, K., Humphries, M. D., & Redgrave, P. (2006). A robot model of the basal ganglia: behavior and intrinsic processing. Neural Networks, 19(1), 31-61.
