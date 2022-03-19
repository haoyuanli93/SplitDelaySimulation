# Introduction
This is the repo as the code base for the split-delay commissioning paper. 

# Requirement
I will not publish this repo as a python package since
this repo has complicated dependence and relies on an old package 
which is no longer maintained.

People who would like to use this to simulate the SD device,
need to prepare the simulation environment.

The critical python dependence for this package is the following:

1. python >= 3.6
2. Numba

With these two, one can use the module XRaySimulation included in this repo.

If one would like to run the energy efficiency calculation,
one also needs the following package

3. pyculib

With this package, one can use the CrystalDiff module included in this repo. 

The other dependence is routine and relatively to solve.
Besides, one needs a Nvidia GPU newer than GTX1060 to run this simulation.

# Unit and Definition
The basic unit of this simulation is um, fs, and keV. 
All parameters feed into this simulation or quantities calculated
from this simulation should follow this unit.

Another thing to be mentioned is the "wave number". The wavenumber is defined to be
    pi * 2 / wavelength.

# Warning
I have tried to reproduce this simulation with my latest X-ray simulation
package. However, I failed. It's quite complicated to complete the translation.
Therefore, I gave up.

Instead, I copy some file from my new repo XRaySimulation to the CrystalDiff folder
and just managed to update this simulation a little bit. Therefore, the code in the 
following files: "misc.py", "multidevice.py"
and "SplitDelayCrystalModel.py",
is just some external code that does not necessarily work well with the other code.

# Future
Unless absolutely necessary, I'll not update or reproduce
this simulation anymore.
This repo serves for the simulation in one of my papers to be published.
I'll keep it as it was.
