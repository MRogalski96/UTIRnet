# UTIRnet
UTIRnet is a supervised universal convolutional neural network for twin-image effect removal in digital in-line holographic microscopy (DIHM)

# Contents and codes
- main_example.m - main code with examples how to: (1) generate network training data, (2) train UTIRnet and (3) reconstruct holograms with UTIRnet <br>
- AS_propagate_p.m - function for optical field propagation with angular spectrum (AS) method <br>
- GenerateDataset.m - function for generating a whole dataset for UTIRnet training <br>
- GenerateHologram.m - function for generating a single hologram and a pair of input-target images for network training <br>
- NetworkArchitecture.m - function for creating CNN architecture <br>
- UTIRnetReconstruction.m - function for reconstructing holograms with the UTIRnet <br>
- ./Holograms - directory where experimental holograms may be stored (see Experimental data section) <br>
- ./Networks - directory with two trained UTIRnets that we employed in our article (see Cite as section) <br>

# How does it work
Follow the steps in main_example.m code to generate synthetic training data and then to train UTIRnet network for a specified system parameters (wavelength, pixel size, magnification, sample-focus plane distance (or sample-camera distance in lensless DIHM system)). Then, generated network (composed from CNN_A and CNN_P networks) along with AS propagation may be used to reconstruct holograms without twin-image effect. 

# Experimental data
Our experimental data (holograms and reference reconstructions) may be found at: <br>
M. Rogalski, P. Arcab, L. Stanaszek, V. Micó, C. Zuo, and M. Trusiak, “Physics-driven universal twin-image removal network for digital in-line holographic microscopy - dataset,” Jun. 2023, doi: 10.5281/ZENODO.8059636. <br>
https://zenodo.org/record/8059636

# Cite as
M. Rogalski, P. Arcab, L. Stanaszek, V. Micó, C. Zuo, and M. Trusiak, “Physics-driven universal twin-image removal network for digital in-line holographic microscopy,” Opt. Express, vol. 32, no. 1, p. 742, Jan. 2024, doi: 10.1364/OE.505440.

# Created by
Mikołaj Rogalski, <br>
mikolaj.rogalski.dokt@pw.edu.pl <br>
Institute of Micromechanics and Photonics, <br>
Warsaw University of Technology, 02-525 Warsaw, Poland <br>
