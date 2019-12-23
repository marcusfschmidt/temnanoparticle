## Example code for plotting from relaxed subsystems.

from __future__ import print_function
from Function_Collections import *
from ase.visualize import view
from gpaw import GPAW, PW
from ase.io import read,write
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

#Define parameters
metal = 'Au' #Define the metal comprising the substrate/nanoparticle
lc = 4.08000 #Lattice constant
surfaces = [(1, 0, 0),(1, 1, 1)]
layers = [7,6]
site = 'OnTop'    #Either 'OnTop' or 'Hollow'. Should be 'OnTop' for 100% coverages.
coverage = 1/4 #Desired coverage. Must be 1/n where n is whole number. 
cell_height=6*lc  #The height of the subsystem unit cell.
subsys_height = 3  #Height of subsystem - no. of atoms.
unmasked_layers = 1 #The number of layers in the gold subsystem (from the top) that is allowed to move during DFT relaxation.
E_cut=400 #Cutoff energy for DFT PW.
kpoints=(10,10,1) #k-points in first Brillouin zone.
ads_cutoff=-2.0
ads111=-3
ads100=-3

adsorbate_ethan = read('ethanethiolate.traj') #Molecule to be adsorbed on top. 

subsys111_relaxed_ethan=read('relax111_ethanethiolate_25_Au.traj')
subsys100_relaxed_ethan=read('relax100_ethanethiolate_25_Au.traj')
relax=False

Nanoparticle_ethan,indexed_surfaces_ethan,subsys111_ethan,subsys100_ethan = Nanoparticle_Builder(metal,lc,surfaces,layers,adsorbate_ethan,site,coverage,cell_height,subsys_height,unmasked_layers,E_cut,kpoints,ads_cutoff,ads111,ads100,subsys111_relaxed_ethan,subsys100_relaxed_ethan,relax)



#PyQSTEM parameters
temsurface = [1,1,1] # Define the surface for which to investigate with TEM. Will typically be [1,0,0] or [1,1,1].
tilt=[1] #The tilt angle (degrees) around the x-axis. This tilts the system "towards" the image plane.
plane_grid=(300,300) #Set the plane wave grid for the TEM electron wave solution.
num_slices=50 #Set the number of slices for the multislice algorithm.
resample=0.2 # Resample rate in Angstrom/pixel.
dose=[5*10**3] # Electron dose at image plane in electrons/Angstrom squared. This is essentially a SNR-parameter.
v0=[60] #Set the electron beam voltage ([keV]).
defocus=[['Placeholder',-45],['Placeholder',-28]] #Set the defocus (in [Å]) for the system. Must be either ['Scherzer',n] or [None,n] for: Scherzer+n or just n defocus respectively.
Cs=[2*10**4] #Set the spherical aberrations. 
df=[30] #Set the focal spread (this implements beam energy spread and applies chromatic aberrations).
MTF_param=([1,0,0.5,2.3]) #Define MTF parameters for the detector. Alternatively, set MTF_param=None.
                          #We use the parametrized form of the MTF presented in Quantitative Image Simulation and Analysis of Nanoparticles".
                          #and define parameters from an experimental fit presented in this Ph.d. thesis. 
rotation=[4] #Define the rotation of the Nanoparticle in degrees.
blur = [1]


#figimgsnr, axs = plt.subplots(len(rotation),3,figsize=(10,int(5*int(len(rotation)))),dpi=200)

snr_data_ethan=[]
snrdb_data_ethan=[]
snrarr_data_ethan=[]

figimgsnr, axs = plt.subplots(1,2,figsize=(7,3),dpi=200,sharey='row')


for i in range(len(defocus)):
    temparams = [tilt[0],plane_grid,num_slices,resample,dose[0],v0[0],defocus[i],Cs[0],df[0],MTF_param,blur[0],temsurface,rotation[0]]
    snrarr_ethan,nnrarr_ethan,snr_ethan,snrdb_ethan,img_snr_ethan,img_reference_snr_ethan,img_wave_snr_ethan,img_wave_reference_snr_ethan=snrmaster(Nanoparticle_ethan,metal,subsys100_ethan,subsys111_ethan, temparams, 1, indexed_surfaces_ethan,plot=False,debug_box=False,prnt=True)
    
    axs[i].imshow(img_snr_ethan.T,extent=img_wave_snr_ethan.get_extent(),cmap='gray',interpolation='nearest',origin='lower')
    axs[i].set_title('$C_1 = {}$ Å'.format(defocus[i][1]),fontsize=10)
    axs[i].set_xlabel('$x\;\, [Å]$')

axs[0].set_ylabel('Ethanetiolate \n$y\;\, [Å]$')


plt.tight_layout()