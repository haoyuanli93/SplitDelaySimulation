# Introduction
This is the repo for the simulation of the 
grating-based amplitude-splitting delay line.

# Requirement
I will not publish this repo as a python package since
I do not think this package will be popular among users.

For people who would like to use this to simulate the SD device,
please prepare the simulation environment.

The critical python dependence for this package is the following:

1. python >= 3.6
2. Numba
3. pyculib

The other dependence is routine and super easy to solve.
Besides, one needs an Nvidia GPU to run this simulation with this package.

# Unit and Definition
The basic unit of this simulation is um, fs, and keV. 
All parameters feed into this simulation or quantities calculated
from this simulation should follow this unit.

Another thing to be mentioned is the "wave number".
Aside from file "xraysimulationutil.py", the wavenumber is defined to be
    pi * 2 / wavelength.

In file "xraysimulationutil.py", the wavenumber is defined to be 
    1 / wavelength.

I might fix this sometime in the future. It is not guaranteed that I'll do that.

# Warning
I have tried to reproduce this simulation with my latest X-ray simulation
package. However, I failed. It's quite complicated to complete the translation.
Therefore, I gave up.

Instead, I copy some file from my new repo XRaySimulation to the CrystalDiff folder
and just managed to update this simulation a little bit. Therefore, the code in the 
following files: "xraysimulationutil.py", "misc.py", "multidevice.py"
and "SplitDelayCrystalModel.py",
is just some external code that does not necessarily work well with the other code.

# Future
Unless absolutely necessary, I'll not update or reproduce
this simulation anymore.
This repo serves for the simulation in one of my papers to be published.
I'll keep it as it was.
