from __future__ import print_function
from ase import Atoms
from ase.visualize import view
from gpaw import GPAW, PW
from ase.build import fcc100,molecule,add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.optimize import QuasiNewton
from ase.io import Trajectory
from ase.io import read,write
import numpy as np
from ase.cluster.cubic import FaceCenteredCubic
from ase.build import fcc100,fcc111,fcc110
import math
from pyqstem import PyQSTEM
from pyqstem.imaging import CTF
from pyqstem.util import atoms_plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import util

def Nanoparticle_Builder(metal,lc,surfaces,layers,adsorbate,site,coverage,cell_height,subsys_height,unmasked_layers,E_cut,kpoints,ads_cutoff,ads111,ads100,subsys111_relaxed,subsys100_relaxed,relax):
    ###
    ### Function to set up the Nanoparticle base
    ###
    
    #Input pparameters
#     metal: The metal comprising the substrate/nanoparticle. For example 'Au'.
#     lc: Lattice constant. For example 4.0800.
#     surfaces: The surfaces in the NP. For example [(1, 0, 0),(1, 1, 1)].
#     layers: The number of layers from the cetner of the NP to each surface. For example: [6,5]
#     adsorbate: The molecule to be adsorbed on top. Should be an object which ase can read. For example: molecule('CO')
#     site: Either 'OnTop' or 'Hollow'. Should be 'OnTop' for 100% coverages.
#     coverage: Desired coverage. Must be 1/n where n is whole number. 
#     cell_height: The height of the subsystem unit cell. For example 6*lc.
#     subsys_height: Height of subsystem - no. of atom  layers. For example: 4. If reading the relaxed subsystem from a traj file MAKE SURE THE HEIGHTS MATCH.
#     unmasked_layers: The number of layers in the NP subsystem (from the top) that is allowed to move during DFT relaxation.
#     E_cut: Cutoff energy for DFT PW.
#     kpoints: k-point grid in the first Brillouin zone. 
#     ads_cutoff: The cutoff for adsorption energy. This determines whether adsorption has occurred for a surface or not.
#     ads111: If providing a relaxed subsystem, this is the adsorption energy for the 111 surface.
#     ads100: If providing a relaxed subsystem, this is the adsorption energy for the 100 surface.
#     subsys111_relaxed: This is an ASE object containing the relaxed subsystem for 111. Necessary to skip DFT.
#     subsys100_relaxed: This is an ASE object containing the relaxed subsystem for 100. Necessary to skip DFT.
#     relax: A boolean to determine whether or not DFT calculations have to be run or not. If relax=False, make sure to provide subsys111_relaxed and subsys100_relaxed along with ads111 and ads100.

    if site==None:
        site='OnTop'
    if cell_height==None:
        cell_height=6*lc
    if subsys_height==None:
        subsys_height=3
    if unmasked_layers==None:
        unmasked_layers=1
    if E_cut==None:
        E_cut=200
    if kpoints==None:
        kpoints=(6,6,1)
    if ads_cutoff==None:
        ads_cutoff=0
    
    #The bulk structure is created.
    Nanoparticle = FaceCenteredCubic(metal, surfaces, layers, latticeconstant=lc)
    Reference = FaceCenteredCubic(metal, surfaces, layers, latticeconstant=lc)  #A reference is made for TEM imaging without adsorbates.

    ##Alternative wulff_construction-based setup. This requires N_atoms and surf_energies list to be set up (for gold, use surf_energies=[1.23,1]).
    #Nanoparticle = wulff_construction(metal, surfaces, surf_energies,size=N_atoms, 'fcc',rounding='above', latticeconstant=lc)
    #Reference=wulff_construction(metal, surfaces, surf_energies,size=N_atoms, 'fcc',rounding='above', latticeconstant=lc)

    surf_atoms=[] #This list will contain indeces for each atom that is on a surface of the nanoparticle.
    #Each surface is rotated to the top of the unit cell. The corresponding surface atoms are then the atoms with highest y-position.
    for i in Nanoparticle.get_surfaces():
        Nanoparticle.rotate(i,[0,1,0],center="COU")
        y_max=0
        for j in Nanoparticle:
            if j.position[1]>y_max:
                y_max=j.position[1]
        for j in Nanoparticle:
            if round(j.position[1],2)==round(y_max,2):
                surf_atoms.append(j.index)
        Nanoparticle.rotate([0,1,0],i,center="COU")

    #Now we need to identify the edge atoms. These are the atoms that are part of 2 or more surfaces in the nanoparticle. 
    #Therefore, they will be the atoms that appear more than once in surf_atoms:
    marked=[]
    edge_atoms=[]
    for i in surf_atoms:
        if i in marked:
            edge_atoms.append(i)
        else:
            marked.append(i)
            
    #A subsystem of the bulk is defined. This will be the basis of the DFT calculation. We also define relevant functions to
    #translate between bulk atom coordinates and the subsystem coordinates. 

    def sizesetup(coverage): #Define the function to set up the size of the unit cell.
        invconverage = math.floor(1/coverage)
        for i in range(1,invconverage+1):
            for j in range(1,invconverage+1):
                if j/i**2==coverage:
                    return (i,j)
                    break

    #This function wraps lattice coordinates (i,j) back into the corresponding coordinate in the unit cell.
    def coordreference(i,j,ucx,ucy,direction):
        if direction==1:
            ri = i%ucx
            ry = j%ucy
        if direction==3:
            #If the unit cell is not orthogonal:
            if ucx%2!=0 and ucy%2!=0:
                ri = i%ucx
                ry = j%ucy            

            #If the unit cell is orthogonal:
            else:
                i = i+j/2-(j%ucy)/2 #Moving along j also corresponds to movement along i.
                ri = i%ucx
                ry = j%ucy
        return (ri,ry)


    ss = sizesetup(coverage)
    ucx = ss[0]
    ucy = ss[0]
    height = subsys_height
    n_adsorbates=ss[1]

    #Check if the requirement for orthogonal unit cell is met:
    if ss[0]%2==0:
        orthogonal=True
    else:
        orthogonal=False

    size = [ucx,ucy,height]  #Size of the FCC bulk structure. 
    
    
    #Set up subsystems for the 111 and 100 directions.
    subsys111 = fcc111(symbol=metal, size=size,a=lc, vacuum=None,orthogonal=orthogonal)
    subsys100=fcc100(symbol=metal, size=size,a=lc, vacuum=None, orthogonal=True)

    
    # Give all atoms in top layer a coordinate
    atomorigo = ucx*ucy*(height-1)
    n = ucx*ucy*height
    subsys_coordinates={}
    i = 0
    j = 0
    for q in range(atomorigo,n):    
        if (i==ucx):
            j += 1
            i = 0    
        subsys_coordinates[q] = [i,j,0]
        i += 1

    #Now we have to find a set of basis vectors describing the subsystem surface. 
    if ss[0]>1:
        #For 111:
        v1_111=subsys111[atomorigo+1].position-subsys111[atomorigo].position
        v1_111=np.array([v1_111[0],v1_111[1],0])
        v2_111=subsys111[atomorigo+ss[0]].position-subsys111[atomorigo].position
        v2_111=np.array([v2_111[0],v2_111[1],0])
        #For 100:
        v1_100=subsys100[atomorigo+1].position-subsys100[atomorigo].position
        v1_100=np.array([v1_100[0],v1_100[1],0])
        v2_100=subsys100[atomorigo+ss[0]].position-subsys100[atomorigo].position
        v2_100=np.array([v2_100[0],v2_100[1],0])
    else:
        v1_111=np.array([0,0,0])
        v2_111=np.array([0,0,0])
        v1_100=np.array([0,0,0])
        v2_100=np.array([0,0,0])

        
    #Now we add adsorbates matching the coverage. They are added along the diagonal in the unit cell.
    if site=='OnTop':
        position100=[0,0,0]
        position111=[0,0,0]
    else:
        position100=1/2*v1_100+1/2*v2_100
        position111=1/2*v1_111+[0,1/2*v2_111[1],0]

    zig=True #If zig is false, then the 111 adsorbate won't zag.
    zags=0
    adsorbate_atom_links=[]#This list will link adsorbates to an atom in the surface.
    for i in range(0,n_adsorbates):
        zig = not zig
        if zig and i>1:
            zags=zags+1
        for j in adsorbate:
            j.tag=i
        scale=ss[0]/n_adsorbates
        adsorbate_atom_links.append([i*scale,i*scale])
        pos111=position111+i*scale*v1_111+i*scale*v2_111-zags*v1_111
        pos100=position100+i*scale*v1_100+i*scale*v2_100
        add_adsorbate(subsys111,adsorbate,height=1*lc,position=(pos111[0],pos111[1]),mol_index=0)
        add_adsorbate(subsys100,adsorbate,height=1*lc,position=(pos100[0],pos100[1]),mol_index=0)


    subsys111.set_cell([subsys111.cell[0],subsys111.cell[1],([0.,0.,cell_height])])
    subsys100.set_cell([subsys100.cell[0],subsys100.cell[1],([0.,0.,cell_height])])


    #Create vectors with initial coordinates for each atom. Offsets from these coordinates are found after the DFT is run.
    x111_i=[]
    y111_i=[]
    z111_i=[]
    x100_i=[]
    y100_i=[]
    z100_i=[]
    for i in range(0,n):
        x111_i.append(subsys111[i].position[0])
        y111_i.append(subsys111[i].position[1])
        z111_i.append(subsys111[i].position[2])
        x100_i.append(subsys100[i].position[0])
        y100_i.append(subsys100[i].position[1])
        z100_i.append(subsys100[i].position[2])
        
    
    if relax:
        #A subsystem of the bulk is defined. This will be the basis of the DFT calculation. We also define relevant functions to
        #translate between bulk atom coordinates and the subsystem coordinates. 

        ss = sizesetup(coverage)
        ucx = ss[0]
        ucy = ss[0]
        height = subsys_height
        n_adsorbates=ss[1]

        #Check if the requirement for orthogonal unit cell is met:
        if ss[0]%2==0:
            orthogonal=True
        else:
            orthogonal=False

        size = [ucx,ucy,height]  #Size of the FCC bulk structure. Preferably size_z is odd.
        
        #Redefine subsystem for energy calculations:
        subsys111 = fcc111(symbol=metal, size=size,a=lc, vacuum=None,orthogonal=orthogonal)
        subsys100=fcc100(symbol=metal, size=size,a=lc, vacuum=None, orthogonal=True)

        # Give all atoms in top layer a coordinate
        atomorigo = ucx*ucy*(height-1)
        n = ucx*ucy*height
        subsys_coordinates={}
        i = 0
        j = 0
        for q in range(atomorigo,n):    
            if (i==ucx):
                j += 1
                i = 0    
            subsys_coordinates[q] = [i,j,0]
            i += 1

        # Calculate system energies:   
        energyfile = open('energies-%s-%s.txt' % (str(coverage),"butanethiolate_hollow"), 'w')
        energies = {}
        for i in ['adsorbate', 'subsys111', 'subsys100']:
           system = globals()[i].copy()
           if i=='adsorbate':
               system.center(vacuum=5)
               system.set_pbc((1,1,0))
           else:
               system.set_cell([system.cell[0],system.cell[1],([0.,0.,cell_height])])

           calc = GPAW(mode=PW(E_cut),kpts=kpoints,xc='BEEF-vdW',txt='energy-%s.txt' % i)
           system.set_calculator(calc)

           energy = system.get_potential_energy()
           energies[i] = energy 

        #Now we have to find a set of basis vectors describing the subsystem surface. 
        if ss[0]>1:
            #For 111:
            v1_111=subsys111[atomorigo+1].position-subsys111[atomorigo].position
            v1_111=np.array([v1_111[0],v1_111[1],0])
            v2_111=subsys111[atomorigo+ss[0]].position-subsys111[atomorigo].position
            v2_111=np.array([v2_111[0],v2_111[1],0])
            #For 100:
            v1_100=subsys100[atomorigo+1].position-subsys100[atomorigo].position
            v1_100=np.array([v1_100[0],v1_100[1],0])
            v2_100=subsys100[atomorigo+ss[0]].position-subsys100[atomorigo].position
            v2_100=np.array([v2_100[0],v2_100[1],0])
        else:
            v1_111=np.array([0,0,0])
            v2_111=np.array([0,0,0])
            v1_100=np.array([0,0,0])
            v2_100=np.array([0,0,0])

        #Now we add adsorbates matching the coverage. They are added along the diagonal in the unit cell.
        if site=='OnTop':
            position100=[0,0,0]
            position111=[0,0,0]
        else:
            position100=1/2*v1_100+1/2*v2_100
            position111=1/2*v1_111+[0,1/2*v2_111[1],0]

        zig=True #If zig is false, then the 111 adsorbate won't zag.
        zags=0   #This is the number of zags that have been performed.
        adsorbate_atom_links=[]#This list will link adsorbates to an atom in the surface.
        for i in range(0,n_adsorbates):
            zig = not zig
            if zig and i>1:
                zags=zags+1
            for j in adsorbate:
                j.tag=i
            scale=ss[0]/n_adsorbates
            adsorbate_atom_links.append([i*scale,i*scale])
            #pos111=position111+((i*scale)/2)*v1_111+[0,i*scale*v2_111[1],0]
            pos111=position111+i*scale*v1_111+i*scale*v2_111-zags*v1_111
            pos100=position100+i*scale*v1_100+i*scale*v2_100
            add_adsorbate(subsys111,adsorbate,height=1*lc,position=(pos111[0],pos111[1]),mol_index=0)
            add_adsorbate(subsys100,adsorbate,height=1*lc,position=(pos100[0],pos100[1]),mol_index=0)


        subsys111.set_cell([subsys111.cell[0],subsys111.cell[1],([0.,0.,cell_height])])
        #subsys110.set_cell([subsys110.cell[0],subsys110.cell[1],([0.,0.,cell_height])])
        subsys100.set_cell([subsys100.cell[0],subsys100.cell[1],([0.,0.,cell_height])])


        #Create vectors with initial coordinates for each atom. Offsets from these coordinates are found after the DFT is run.
        x111_i=[]
        y111_i=[]
        z111_i=[]
        x100_i=[]
        y100_i=[]
        z100_i=[]
        for i in range(0,n):
            x111_i.append(subsys111[i].position[0])
            y111_i.append(subsys111[i].position[1])
            z111_i.append(subsys111[i].position[2])
            x100_i.append(subsys100[i].position[0])
            y100_i.append(subsys100[i].position[1])
            z100_i.append(subsys100[i].position[2])

        #The calculator is set.
        calc111 = GPAW(mode=PW(E_cut),kpts=kpoints, xc='BEEF-vdW',txt='calc111.out')
        subsys111.set_calculator(calc111)
        #The bottom layers of the subsystem are masked such that these atoms do not move during QuasiNewton minimization/relaxation.
        mask = [i.tag > unmasked_layers and i.symbol==metal for i in subsys100]
        fixedatoms=FixAtoms(mask=mask)
        subsys111.set_constraint(fixedatoms)
        #The subsystem is relaxed until all forces on atoms are below fmax=0.02. 
        relax = QuasiNewton(subsys111, trajectory='relax111.traj')
        relax.run(fmax=0.02)

        #The calculator is set.
        calc100 = GPAW(mode=PW(E_cut),kpts=kpoints, xc='BEEF-vdW',txt='calc100.out')
        subsys100.set_calculator(calc100)
        #The bottom layer of the subsystem is masked such that these atoms do not move during QuasiNewton minimization/relaxation.
        subsys100.set_constraint(fixedatoms)
        #The subsystem is relaxed until all forces on atoms are below fmax=0.02. 
        relax = QuasiNewton(subsys100, trajectory='relax100.traj')
        relax.run(fmax=0.02)

        ## Calculate new energies
        for i in ['subsys111', 'subsys100']:
           system = globals()[i]
           energy = system.get_potential_energy()
           energies["relax"+i[6:]] = energy
        e_bond = {}
        e_bond['subsys111'] = energies['relax111'] - energies['subsys111'] - n_adsorbates*energies['adsorbate']
        e_bond['subsys100'] = energies['relax100'] - energies['subsys100'] - n_adsorbates*energies['adsorbate']
        print(energies,e_bond, file=energyfile)
        energyfile.close()
        
        ads111=e_bond['subsys111']
        ads100=e_bond['subsys100']
    
    if subsys100_relaxed!=None:
        subsys100=subsys100_relaxed
    
    if subsys111_relaxed!=None:
        subsys111=subsys111_relaxed
    
    #Check if adsorption has occurred. If not, remove adsorbates from corresponding subsystem.
    if ads111>=ads_cutoff:
        subsys111_prov=subsys111.copy()
        subsys111=Atoms()
        for i in subsys111_prov:
            if i.symbol==metal:
                subsys111.append(i)
    
    if ads100>=ads_cutoff:
        subsys100_prov=subsys100.copy()
        subsys100=Atoms()
        for i in subsys100_prov:
            if i.symbol==metal:
                subsys100.append(i)
    
    #Now to find offsets for each atom in the subsystems after DFT:
    x111_offset=[]
    y111_offset=[]
    z111_offset=[]
    x100_offset=[]
    y100_offset=[]
    z100_offset=[]
    for i in range(0,n):
        x111_offset.append(subsys111[i].position[0]-x111_i[i])
        y111_offset.append(subsys111[i].position[1]-y111_i[i])
        z111_offset.append(subsys111[i].position[2]-z111_i[i])
        x100_offset.append(subsys100[i].position[0]-x100_i[i])
        y100_offset.append(subsys100[i].position[1]-y100_i[i])
        z100_offset.append(subsys100[i].position[2]-z100_i[i])

        
    
    
    # Define dictionary of indexed surfaces for use in TEM
    indexed_surfaces= {}

    #Sadly, we now have to do all the rotations again:
    surface_coordinates={} #For convenience this dictionary will contain v1,v2 coordinates for all the surfaces - indexed by their
    #respective origo atom's index.
    for i in Nanoparticle.get_surfaces():
        surface=[] #This list will contain the atoms on the specific surface at the top.
        Nanoparticle.rotate(i,[0,1,0],center="COU")
        y_max=0
        for j in Nanoparticle:
            if j.position[1]>y_max and j.symbol==metal:
                y_max=j.position[1]
        for j in Nanoparticle:
            if round(j.position[1],2)==round(y_max,2) and j.symbol==metal:
                surface.append(j.index)
        #Now surface contains the indeces of atoms of the specific surface that is at the top in this rotation.
        direction=abs(i[0])+abs(i[1])+abs(i[2]) #This number determines the surface direction family - 100, 110, 111.

        #Define a dictionary with all indeces of atoms of surface i
        indexed_surfaces[tuple(i)] = surface

        
        #Find maximum z and x values for the surface:
        x_max=0
        z_max=0
        for k in surface:
            if Nanoparticle[k].position[0]>x_max:
                x_max=Nanoparticle[k].position[0]
            if Nanoparticle[k].position[2]>z_max:
                z_max=Nanoparticle[k].position[2]
        x_min=x_max
        z_min=z_max
        for k in surface:
            if Nanoparticle[k].position[0]<x_min:
                x_min=Nanoparticle[k].position[0]
            if Nanoparticle[k].position[2]<z_min:
                z_min=Nanoparticle[k].position[2]
        bot_row=[] #This will contain the indeces of the bottom (low z) row of atoms.
        #Find the atoms in the bottom row:
        for k in surface:
            if round(Nanoparticle[k].position[2],1)==round(z_min,1):
                bot_row.append(k)
                
        #Find the atom in the corner of the surface:
        corner_atom=bot_row[0]
        for k in bot_row:
            if Nanoparticle[k].position[0]<Nanoparticle[corner_atom].position[0]:
                corner_atom=k
        distance_1=2*lc #The distance between corner_atom and the nearest neighbour is at least smaller than this.

        neighbour_1="d" #placeholder for neighbour 1.
        neighbour_2="d" #placeholder for neighbour 2.

        ## Find the unit cell neighbours to corner_atom.
        v1=[]
        v2=[]
        for k in surface:
            if k!=corner_atom and k in edge_atoms:  #The v1-axis should lie along some edge.
                if np.linalg.norm(Nanoparticle[k].position-Nanoparticle[corner_atom].position)<distance_1:
                    distance_1=np.linalg.norm(Nanoparticle[k].position-Nanoparticle[corner_atom].position)
                    neighbour_1=k

        #Construct the first basis vector for the surface using the first nearest neighbour coordinate.
        v1=Nanoparticle[neighbour_1].position-Nanoparticle[corner_atom].position
        v1[1]=0 #The y-coordinate of the surface basis vector is set to 0. 

        # To find the second neighbour, we have to align the v1 vector with the x-axis:
        Nanoparticle.rotate(v1,[1,0,0],center='COU')
        for k in surface:
            if k!=corner_atom and k!=neighbour_1:
                dist_vector=Nanoparticle[k].position-Nanoparticle[corner_atom].position
                # We require that the angle between dist_vector and v1 is <=90. 
                if math.acos(round(np.dot(dist_vector,v1)/(np.linalg.norm(dist_vector)*np.linalg.norm(v1)),5))<=90:
                    # We check for a dist_vector which corresponds to one of the lattice vectors defined for the subsystem.
                    if direction==1:
                        if round(dist_vector[0],5)==round(v2_100[0],5) and round(dist_vector[2],5)==round(v2_100[1],5):
                            neighbour_2=k
                        if round(dist_vector[0],5)==round(v2_100[0],5) and round(dist_vector[2],5)==-round(v2_100[1],5):
                            neighbour_2=k
                    if direction==3:
                        if round(dist_vector[0],5)==round(v2_111[0],5) and round(dist_vector[2],5)==round(v2_111[1],5):
                            neighbour_2=k
                        if round(dist_vector[0],5)==round(v2_111[0],5) and round(dist_vector[2],5)==-round(v2_111[1],5):
                            neighbour_2=k

        # Rotate the system back after finding the second neighbour.
        Nanoparticle.rotate([1,0,0],v1,center='COU')

        #Construct the second basis vector for the surface using the nearest neighbour coordinate.
        v2=Nanoparticle[neighbour_2].position-Nanoparticle[corner_atom].position

        v2[1]=0 #The y-coordinate of the surface basis vector is set to 0. 

        Transform_matrix=np.linalg.inv(np.array([v1,v2,[0,1,0]]).transpose()) #This transforms x-y-z coordinates to v1-v2-y.
        #Now to find v1,v2-coordinates for all the atoms in the surface and replace them with atoms from DFT subsystem accordingly:

        surface.sort
        for k in surface:
            if k not in edge_atoms:
                flag = False #Flag to determine wether the z-axis was flipped.

                #Find the coordinate of the atom in v1,v2-basis. 
                coordinate=np.round(Transform_matrix.dot(Nanoparticle[k].position-Nanoparticle[corner_atom].position),0)

                #We want the origo of the surface system to be off the surface edge. Therefore, we translate the coordinates:
                coordinate[0]=coordinate[0]-1
                coordinate[1]=coordinate[1]-1

                reference=coordreference(coordinate[0],coordinate[1],ucx,ucy,direction) #This references the matching atom in the subsystem

                #The system is rotated again such that the bottom row of atoms lie along the x-direction.
                Nanoparticle.rotate(v1,[1,0,0],center="COU")
                #Check if v2 is in the positive z-direction. If not, rotate the system 180 degrees:
                if Nanoparticle[neighbour_2].position[2]-Nanoparticle[corner_atom].position[2]<0:
                    Nanoparticle.rotate([0,0,-1],[0,0,1],center='COU')
                    flag = True #Flag is set indicating the system has been flipped.
                for l in subsys_coordinates:
                    if subsys_coordinates[l]==[reference[0],reference[1],0]: #This atom in the subsystem matches the atom in the Nanoparticle surface.
                        if direction==1:
                            #Apply the corresponding offset from the 100 list:
                            Nanoparticle[k].position=Nanoparticle[k].position+[x100_offset[l],z100_offset[l],y100_offset[l]]
                        if direction==3:
                            #Apply the corresponding offset from the 111 list:
                            Nanoparticle[k].position=Nanoparticle[k].position+[x111_offset[l],z111_offset[l],y111_offset[l]]
                if [reference[0],reference[1]] in adsorbate_atom_links: #Checks if the subsystem reference atom is linked to an adsorbate molecule.
                    for u in range(0,len(adsorbate_atom_links)): #Find the linked adsorbate molecule tag (it's the index of adsorbate_atom_links)
                        if adsorbate_atom_links[u] == [reference[0],reference[1]]: #Check if the reference coordinates exists in the list of linked atoms coordinates.
                            adsorbate_tag=u #This is the tag assigned to the adsorbate that is linked to the reference atom.
                    #Calculate the reference atom's index in the subsystem (from the v1/v2-coordinates).
                    tagged_atom_index=int(atomorigo+((adsorbate_atom_links[adsorbate_tag][1])%ucy)*ucx+adsorbate_atom_links[adsorbate_tag][0]%ucx)
                    if direction==1:
                        for m in subsys100:
                            if m.symbol!=metal and m.tag==adsorbate_tag: #Single out the adsorbate with the correct tag.
                                m_old_position=m.position.copy() #Save the adsorbate's original position.
                                m.position=m.position-subsys100[tagged_atom_index].position #Calculate the vector from the reference atom to the adsorbate.
                                #Now calculate and set the position to that which the adsorbate atom should have in the Nanoparticle system:
                                m.position=[m.position[0]+Nanoparticle[k].position[0],m.position[2]+Nanoparticle[k].position[1],m.position[1]+Nanoparticle[k].position[2]]
                                Nanoparticle.append(m) #Finally, add the adsorbate.
                                m.position=m_old_position #Reset the subsystem (set the adsorbate position to it's old, saved value).
                    if direction==3:
                        #Do exactly the same below as for the 100 surface above:
                        for m in subsys111:
                            if m.symbol!=metal and m.tag==adsorbate_tag:
                                m_old_position=m.position.copy()
                                m.position=m.position-subsys111[tagged_atom_index].position
                                m.position=[m.position[0]+Nanoparticle[k].position[0],m.position[2]+Nanoparticle[k].position[1],m.position[1]+Nanoparticle[k].position[2]]
                                Nanoparticle.append(m)
                                m.position=m_old_position


                #Check if the z-axis was flipped. If it was, flip it back:
                if flag:
                    Nanoparticle.rotate([0,0,1],[0,0,-1],center='COU')

                #The system is then rotated back.
                Nanoparticle.rotate([1,0,0],v1,center="COU") #First the x-axis alignment. 

        Nanoparticle.rotate([0,1,0],i,center="COU") #Now rotate the surface back to it's original direction.

        
        
    ## Find the actual coverage of the Nanoparticle:
    ## This procedure is outdated, but it still works. 

    #First we need to count the number of surface atoms (including edges). First we turn surf_atoms into a dictionary and back into
    # a list in order to remove duplicate values (edge atoms should only be counted once):
    surf_atoms = list(dict.fromkeys(surf_atoms))

    #The number of surf atoms is then:
    n_surf_atoms=len(surf_atoms)

    #Now to count the number of adsorbed molecules. First we count the number of non-gold atoms:
    non_gold_atoms=0
    for i in Nanoparticle:
        if i.symbol!=metal:
            non_gold_atoms+=1
    #Then we count the number of atoms in the adsorbate molecules:
    adsorbate_size=len(adsorbate)

    #The number of adsorbed molecules:
    n_adsorbate=non_gold_atoms/adsorbate_size

    #The actual coverage is then:
    actual_coverage=n_adsorbate/n_surf_atoms
    
    
    
    
    Nanoparticle.center(vacuum=4) #Center the nanoparticle for TEM imaging.
    
    return Nanoparticle,indexed_surfaces,subsys111,subsys100 #Depending on what is needed, return the different objects.

## Define PyQSTEM functions related to TEM simulation.

#Define a TEM imaging function:
def IMG_sim(system,temparams):
    qstem = PyQSTEM('TEM') # Initialize a QSTEM simulation of TEM image.
    #Read the TEM parameters (given in a temparams list.)   
    tilt,plane_grid,num_slices,resample,dose,v0,defocus,Cs,df,MTF_param,blur,temsurface,rotation = temparams
    
    #Define defocus function to calculate Scherzer defocus (and offset from Scherzer defocus):
    def defocusfunc(focus):
        result=[0,focus[1]]
        if focus[0]=='Scherzer':
            prov_sys=molecule('H',vacuum=2)
            qstem.set_atoms(prov_sys)
            qstem.build_wave('plane',v0,plane_grid)
            prov_wave=qstem.get_wave()
            lam=prov_wave.wavelength
            result[0]=-1.2*math.sqrt(lam*Cs)
        else:
            result[0]=0
        return sum(result)
    
    defocus=defocusfunc(defocus)
    
    # Rotate temsurface to the top:
    system.rotate(temsurface,[0,1,0],center='COU')
    
    # Apply rotation to system and reference system.
    system.rotate(rotation,v=[0,1,0],center="COU")

    # Apply the tilt
    system.rotate(tilt,[1,0,0],center="COU")
    
    qstem.set_atoms(system) #Set the atoms for the first PyQSTEM simulation.
    qstem.build_wave('plane',v0,plane_grid) #Build the plane wave on the defined grid.
    wave=qstem.get_wave() #Save the plane wave in Python.
    qstem.build_potential(num_slices) #Build the electrostatic potential from the system using tabulated data for each atom. From Rez et al.: "Dirac-Fock calculations of X-ray scattering factors and contributions to the mean inner potential for electron scattering".
    potential=qstem.get_potential_or_transfunc() #Save the potential in Python. Have a look at it. 
    qstem.run() #Run the qstem simulation.
    wave=qstem.get_wave() #Get the exit wave.
    ctf = CTF(defocus=defocus,Cs=Cs,focal_spread=df) #Define a contrast transfer function from relevant parameters.
    img_wave=wave.apply_ctf(ctf) #Apply the CTF to the exit wave and save the resulting image wave.
    img=img_wave.detect(dose=dose,resample=resample,MTF_param=MTF_param,blur=blur) # Apply the detector PSF and resample the image. 
    
    # Remove the tilt of the system.
    system.rotate(-tilt,[1,0,0],center="COU")
    
    # Rotate the system back.
    system.rotate(-rotation,v=[0,1,0],center="COU")
    
    # Rotate temsurface back.
    system.rotate([0,1,0],temsurface,center='COU')
    
    return img,img_wave,defocus

#Define function to calculate the S/N ratio
#SNR value is the mean based on nmean SNR calculations.
def snrmaster(Nanoparticle,metal,subsys100,subsys111, temparams, nmean, indexed_surfaces,plot,debug_box,prnt):
    
    #Produce the Nanoparticle SNR reference cell
    Nanoparticlesnr = Atoms()
    for i in Nanoparticle:
        Nanoparticlesnr.append(i)
    Nanoparticlesnr.set_cell(Nanoparticle.cell)
    Nanoparticlesnr.center()

    #Same procedure for the SNR reference cell
    Referencesnr = Atoms()
    for i in Nanoparticle:
        if i.symbol==metal:
            Referencesnr.append(i)
    Referencesnr.set_cell(Nanoparticle.cell)
    Referencesnr.center()
    
    
    #Read the TEM parameters    
    tilt,plane_grid,num_slices,resample,dose,v0,defocus,Cs,df,MTF_param,blur,temsurface,rotation = temparams
    
    SNRparams=[0,plane_grid,num_slices,resample,dose,v0,defocus,Cs,df,MTF_param,blur,[0,1,0],0]
    
    qstem = PyQSTEM('TEM')
    
    #Rotate systems to the surface we wish to investigate.
    Referencesnr.rotate(temsurface,[0,1,0],center='COU')
    Nanoparticlesnr.rotate(temsurface,[0,1,0],center='COU')
    
    #Rotate the system according to rotation parameter:
    Referencesnr.rotate(rotation,[0,1,0],center='COU')
    Nanoparticlesnr.rotate(rotation,[0,1,0],center='COU')
    
    
    #Tilt NP and reference S/N in order to match the system we visualise and the system we calculate S/N from.
    Nanoparticlesnr.rotate(tilt,[1,0,0],center="COU")
    Referencesnr.rotate(tilt,[1,0,0],center="COU")

    
    # First setup of the TEM objects for the NP and the reference system.
    # The setup happens before the loop designed to get an average since we need an img_wave object to define the S/N box.
    img_snr,img_wave_snr,defocus_value=IMG_sim(Nanoparticlesnr,SNRparams)
    
   
    img_reference_snr,img_wave_reference_snr,defocus_value=IMG_sim(Referencesnr,SNRparams)
    

    #Find the height and width of the box within which to calculate the value of the signal.
    def heightbox():
        dire = abs(temsurface[0])+abs(temsurface[1])+abs(temsurface[2])
        if (dire == 1):
            subsys = subsys100
        if (dire == 3):
            subsys = subsys111
        zmax = 0
        xmax = 0
        ymax = 0
        zmaxmetal = 0
        xmaxmetal = 0
        ymaxmetal = 0
        
        #Maximum x, y and z values are found for the subsystem:
        xmin = 0
        ymin = 0
        for i in subsys:
            if(i.position[2]>zmax):
                zmax = i.position[2]
            if(i.position[2]>zmaxmetal and i.symbol == metal):
                zmaxmetal = i.position[2]
            if abs(i.position[0])>xmax:
                xmax=abs(i.position[0])
            if abs(i.position[0])>xmaxmetal and i.symbol == metal:
                xmaxmetal=abs(i.position[0])
            if abs(i.position[1])>ymax:
                ymax=abs(i.position[1])
            if abs(i.position[1])>ymax and i.symbol == metal:
                ymaxmetal=abs(i.position[1])
            if i.position[0]<xmin and i.symbol != metal:
                xmin=i.position[0]
            if i.position[1]<ymin and i.symbol !=metal:
                ymin=i.position[1]
        
        #Determine if adsorbate atoms have x-y-coordinates outside the subsystem unit cell.
        xyoffsets=[0]
        if xmax>xmaxmetal:
            xyoffsets.append(xmax-xmaxmetal)
        if ymax>ymaxmetal:
            xyoffsets.append(ymax-ymaxmetal)
        if ymin<0:
            xyoffsets.append(abs(ymin))
        if xmin<0:
            xyoffsets.append(abs(xmin))
            
        #Determine the offset in x,y,z:
        zoffset=zmax-zmaxmetal
        xyoffset=max(xyoffsets)
        
        return zoffset, xyoffset

    #Define a function to determine x,y and z coordinates for the SNR box in the nanoparticle unit cell:
    def coordinatesbox():
        xmax = 0
        height = heightbox()[0]
        ind = indexed_surfaces[tuple(temsurface)][0]
        
        zmax = 0
        for i in indexed_surfaces[tuple(temsurface)]:
            pos = Nanoparticlesnr[i].position[1]
            if(pos>zmax):
                zmax = pos
                
        zmin = zmax
        for i in indexed_surfaces[tuple(temsurface)]:
            pos = Nanoparticlesnr[i].position[1]
            if(pos<zmin):
                zmin = pos
                
        zdif = zmax-zmin
        z = zmax
        height = (height + zdif)*1.2
        
        for i in indexed_surfaces[tuple(temsurface)]:
            pos = Nanoparticlesnr[i].position[0]
            if(pos>xmax):
                xmax = pos
        xmin = xmax
        for i in indexed_surfaces[tuple(temsurface)]:
            pos = Nanoparticlesnr[i].position[0]
            if(pos<xmin):
                xmin = pos
        xmax=xmax+heightbox()[1]
        xmin=xmin-heightbox()[1]
        return (xmin,xmax,z,z+height)

    xmin,xmax,zmin,zmax = coordinatesbox()

    xmin,xmax,zmin,zmax = coordinatesbox()
    

    #Conversion between number of data points and length scale
    dpl = len(img_snr.T[0])/img_wave_snr.get_extent()[1]

    #Convert from lengths to corresponding index values in image array
    xmindata = math.floor(xmin*dpl)
    xmaxdata = math.floor(xmax*dpl)
    zmindata = math.floor(zmin*dpl)
    zmaxdata = math.floor(zmax*dpl)
    
    #Define RMS function.
    def rms(img):
        rms = np.sqrt(np.sum(img*img)/len(img.flat))
        return rms

    #Define SNR function.
    def snrfunc(signal,noise):
        
        #We find the SNR of both the image and the corresponding inverted image. We return the maximum value.
        s1 = signal
        s2 = util.invert(signal)

        n1 = noise
        n2 = util.invert(noise)

        snr1 = rms(s1)/rms(n1)
        snr2 =rms(s2)/rms(n2)

        return max(snr1,snr2)


    zlen = zmaxdata - zmindata
    
    
    
    #Define arrays to contain SNR values.
    snrarr = []
    nnrarr = []
    
    # Make first data point of the average
    imgdif = img_snr-img_reference_snr
    
    
    #Select the data points where signal and noise is calculated from:
    imgcut = imgdif[xmindata:xmaxdata,zmindata:zmaxdata]
    imgcutnoise = imgdif[0:20,0:20]
    
    #If debug_box=True, set the pixel values corresponding to the SNR boxes to 0.
    if debug_box:
        imgdif[xmindata:xmaxdata,zmindata:zmaxdata]=0
        imgdif[0:20,0:20]=0


    imgcut_reference_snr=img_reference_snr[xmindata:xmaxdata,-1-zlen:-1]
    imgcutnoise_reference_snr = img_reference_snr[0:20,0:20]
    
    
    
    #For debugging.
    extentcut = [xmindata,xmaxdata,zmindata,zmaxdata]
   
    snr = snrfunc(imgcut,imgcutnoise)
    snrref = snrfunc(imgcut_reference_snr,imgcutnoise_reference_snr)
    
    snrarr.append(snr)
    nnrarr.append(snrref)
    
    #Loop over nmean 
    
    for i in range(1,nmean):
        
        #Construct qstem objects again for use in the averaging
        img_snr,img_wave_snr,defocus_value=IMG_sim(Nanoparticlesnr,SNRparams)
        img_reference_snr,img_wave_reference_snr,defocus_value=IMG_sim(Referencesnr,SNRparams)


        #calculate SNR and NNR and append to array
        imgdif = img_snr-img_reference_snr

        imgcut = imgdif[xmindata:xmaxdata,zmindata:zmaxdata]
        imgcutnoise = imgdif[0:20,0:20]

        #imgcut_reference_snr=img_reference_snr[xmindata:xmaxdata,zmindata:zmaxdata]
        imgcut_reference_snr=img_reference_snr[xmindata:xmaxdata,-1-zlen:-1]
        imgcutnoise_reference_snr = img_reference_snr[0:20,0:20]

        extentcut = [xmindata,xmaxdata,zmindata,zmaxdata]

        snr = snrfunc(imgcut,imgcutnoise)
        snrref = snrfunc(imgcut_reference_snr,imgcutnoise_reference_snr)

        snrarr.append(snr)
        nnrarr.append(snrref)
        
        
    #Rotate and tilt back
    Nanoparticlesnr.rotate(-tilt,[1,0,0],center="COU")
    Referencesnr.rotate(-tilt,[1,0,0],center="COU")
    
    Referencesnr.rotate(-rotation,[0,1,0],center='COU')
    Nanoparticlesnr.rotate(-rotation,[0,1,0],center='COU')

    Nanoparticlesnr.rotate([0,1,0],temsurface,center='COU')
    Referencesnr.rotate([0,1,0],temsurface,center='COU')
    
    finalsnr = np.average(snrarr)
    finalsnrdb = 10*math.log10(finalsnr)
    finalnnr = np.average(nnrarr)
    finalnnrdb = 10*math.log10(finalnnr)
    
    if(prnt):
        print('Average SNR: '+str(finalsnr))
        print('Average SNR in dB:' +str(finalsnrdb))
        print('Average reference NNR: ' + str(finalnnr))
        print('Average reference NNR in dB: ' +str(finalnnrdb)+'\n')

    
    #If plot=True, plot the different pixel arrays used in the calculation of SNR:
    if(plot):
        
        figimgsnr, axs = plt.subplots(1,3,figsize=(10,7),dpi=300)
        axs[0].imshow(img_snr.T,extent=img_wave_snr.get_extent(),cmap='gray',interpolation='nearest',origin='lower')
        axs[1].imshow(img_reference_snr.T,extent=img_wave_reference_snr.get_extent(),cmap='gray',interpolation='nearest',origin='lower')
        axs[2].imshow(imgdif.T,extent=img_wave_reference_snr.get_extent(),cmap='gray',interpolation='nearest',origin='lower')    
        
        
        figsnr = plt.figure(figsize=(5,4),dpi=200)
        x = np.arange(nmean)
        plt.plot(x,snrarr,label='S/N ratio',linewidth=2,marker='s',color='blue')
        plt.plot(x,nnrarr,label='N/N ratio',linewidth=2,marker='o',color='red')
        plt.axhline(y=np.average(snrarr), xmin=0, xmax=nmean, color='green',linestyle='--',label='Average S/N:\n {:.3} dB'.format(finalsnrdb))
        plt.axhline(y=np.average(nnrarr), xmin=0, xmax=nmean, color='black',linestyle='--',label='Average N/N:\n {:.3} dB'.format(finalnnrdb))
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
        plt.xlabel('Data points')
        plt.ylabel('Ratio value')
        plt.title('Average value of S/N-ratio')
        plt.legend()
    
    
    #Depending on what is needed, return the following objects:
    return snrarr,nnrarr,finalsnr,finalsnrdb,img_snr,img_reference_snr,img_wave_snr,img_wave_reference_snr