#!/usr/bin/python
# Author: Alejandro J. Garza 
# Purpose Compute entropy in solution
import sys
import os
import string
import numpy as np
import math as m
import random
import time
import scipy
from scipy import special


def print_usage():
    print("Usage: ")
    print("python entropy-solution.py imput_file")
    print(" ")
    print("The input file must define the following")
    print("mandatory arguments:")
    print(" solute = (xyz file for solute molecule)")
    print(" solvent = (xyz file for solvent molecule)")
    print(" omega = (Pitzer acentric factor of solvent)")
    print(" rho_solvent = (density of solvent in g/mL)")    
    print(" ")
    print("The following arguments are optional:")
    print(" freq_file = (a file with the vibrational frequencies in Gaussian format)")
    print(" linear = (True/False; whether the molecule is linear; default = False )")
    print(" symmetry = (rotational symmetry number of solute; default = 1)")    
    print(" vol_solute = (Solute volume in A^3)")
    print(" area_solute = (Surface area of solute in A^2)")
    print(" vol_solvent = (Solvent volume in A^3)")
    print(" area_solvent = (Surface area of solvent in A^2)")
    print(" dipole_solute = (dipole moment of solute in Debye)")
    print(" dipole_solvent = (dipole moment of solvent in Debye)")    
    print(" temperature = (temperature in Kelvin; default = 298.15)")
    print(" concentration = (concentration in mol/L of solute; default = 1M)")  
    print(" nu0 = (low frequency threshold in cm^-1; default = 100 cm^-1)")
    print(" ")
    print("Notes:")
    print(" - The format requires each variable to be specified in a separate")
    print("   line in the input file.")
    print(" - Providing the solute volume and surface area can save substantial")
    print("   computation time.")
    print(" - A list of Pitzer acentric factors for common solvents is")
    print("   available at:")
    print("http://www.kaylaiacovino.com/Petrology_Tools/Critical_Constants_and_Acentric_Factors.htm.")    
    return


if (len(sys.argv) != 2):
    print("Error: Wrong number of arguments.")
    print_usage()
    sys.exit()

# Physical Constants
h = (4.13567E-15) # Planck constant in eV*s
k = (8.61733E-5) # Boltzmann constant in eV/K
c = 299792458*100 #speed of light in cm/s
epsilon_0 = 8.9542E-12 # permittivity of vacuum in SI
N_Av = 6.0221409E23 # Avogadro's number
pi = np.pi

# Constants for atoms              
# Atomic masses 
mass = {"H" : 1.008, "D" : 2.014, "He" : 4.003, \
         "C" : 12.011, "N" : 14.007, "O" : 15.999, "Ne" : 20.18, \
         "P" : 30.974, "S" : 32.065, "Cl" : 35.453, "Ar" : 39.948, \
         "Ni" : 58.690, "I" : 126.90, "Kr" : 83.8, \
          "Xe" : 131.29}
# Pitzer accentric factors (for future use)
Pitzer = {"toluene" : 0.263, "water" : 0.344, "iodine" : 0.229}
# Pyykko's covalent radii in pm
r_cov = {"H" : 32, \
         "C" : 75, "N" : 71, "O" : 63, \
         "P" : 111, \
         "Ni" : 110}
# Van der Waals radii in pm
r_vdW = {"H" : 120, "D" : 120, "He" : 140, \
         "C" : 170, "N" : 155, "O" : 152, "Ne" : 154, \
         "P" : 180, "S" : 180, "Cl" : 175, "Ar" : 188, \
         "Ni" : 163, "I" : 198, "Kr" : 202, \
         "Xe" : 216}


# Default values for parameters/thresholds
T = 298.15
P = 100000.0 # Pressure in Pa
concentration = 1 # M
sigma_r = 1 # rotational symmetry number of solute
sigma_r2 = 2 # rotational symmetry number of solvent (default is water)
epsilon_r = 78.54 # Relative permittivity of solvent (default is water)
sigma_0 = None#72.8E-3 # Surface tension coefficient (default is water)
dsigma_dT = None#-0.1514E-3 # Temperature coefficient (default is water)
T_c = None # Solvent critical temperature 
alpha = None #207E-6  # Isobaric thermal volumetric expansion coefficient (default is water)
nu0 = 100 # Threshold for low frequency in Grimme's hindered rotor treatment of vibrations (in cm^-1)
npts_mc = 1000000 # Number of points for Monte Carlo molecular volume evaluation
freq_scal = 1 # Scaling factor of vibrational frequencies
omega = None # Pitzer acentric factor
linear_mol = False
solute_xyz = None
solvent_xyz = None
freq_file = None

# Initialize other optional input variables
rho_solvent = None # density of solvent g/mL
area_solute = 0; vol_solute = 0
area_solvent = 0; vol_solvent = 0
mbb_solute = None; mbb_solvent = None
tot_dipole_solute = None; tot_dipole_solvent = None

# Get parameters/directives from input file
f_directives = open(sys.argv[1],"r")
for line in f_directives.readlines():
    aline = line.split()
    if (len(aline) >= 3):
        aline[0] = aline[0].lower()
        if (aline[0] == 'solvent'):
            solvent_xyz = aline[2]
        elif (aline[0]  == 'solute'):
            solute_xyz = aline[2]
        elif (aline[0]  == 'temperature'):
            T = float(aline[2])
        elif (aline[0] == "pressure"):
            P = float(aline[2])*100000
        elif (aline[0]  == 'symmetry'):            
            sigma_r = float(aline[2])
        elif (aline[0]  == 'symmetry_solvent'):            
            sigma_r2 = float(aline[2])            
        elif (aline[0]  == 'omega'):            
            omega = float(aline[2])
        elif (aline[0]  == 'nu0'):            
            nu0 = float(aline[2])
        elif (aline[0] == "freq_file"):
            freq_file = aline[2]
        elif (aline[0] == "linear"):
            linear_mol = (aline[2].lower() == 'true')
        elif (aline[0]  == 'area_solute'):            
            area_solute = float(aline[2])            
        elif (aline[0]  == 'vol_solute'):            
            vol_solute = float(aline[2])                        
        elif (aline[0]  == 'area_solvent'):            
            area_solvent = float(aline[2])            
        elif (aline[0]  == 'vol_solvent'):            
            vol_solvent = float(aline[2])
        elif (aline[0]  == 'concentration'):
            concentration = float(aline[2])
        elif (aline[0]  == 'dipole_solute'):            
            tot_dipole_solute = float(aline[2])            
        elif (aline[0]  == 'dipole_solvent'):            
            tot_dipole_solvent = float(aline[2])
        elif (aline[0]  == 'rho_solvent'):            
            rho_solvent = float(aline[2])
            print("Solvent Density (g/mL) = ", '%.3f'%rho_solvent)    
        elif (aline[0]  == 'epsilon_r'):            
            epsilon_r = float(aline[2])
            print("Solvent epsilon_r = ", '%.3f'%epsilon_r)    
        elif (aline[0]  == 'alpha'):            
            alpha = float(aline[2])
        elif (aline[0]  == 'mc_points'):            
            npts_mc = int(aline[2])
        elif (aline[0]  == 'sigma_0'):            
            sigma_0 = float(aline[2])
        elif (aline[0]  == 'dsigma_dt'):            
            dsigma_dT = float(aline[2])
        elif (aline[0]  == 't_c'):            
            T_c = float(aline[2])
        elif (aline[0]  == 'dipole_file'):
            dipole_file = aline[2]
        elif (aline[0]  == 'dipole_file_solvent'):
            dipole_file_solvent = aline[2]
        elif (aline[0]  == 'mbb_solute'):
            mbb_solute = float(aline[2])
        elif (aline[0]  == 'mbb_solvent'):
            mbb_solvent = float(aline[2])
        elif (aline[0]  == 'freq_scal'):
            freq_scal = float(aline[2])
            print("freq_scal = ", freq_scal)
            
            
f_directives.close()

if ((omega == None) & (alpha == None)):
    print("Error: Either the Pitzer acentric factor omega or the ")
    print("volumetric expansion coefficent alpha must be defined.")
    sys.exit()
elif (solute_xyz == None):
    print("Error: The solute xyz file name must be given.")
    sys.exit()
elif (solvent_xyz == None):
    print("Error: The solvent xyz file name must be given.")
    sys.exit()

if ((tot_dipole_solvent == None) & (dipole_file_solvent == None)):
    print("Solvent dipole not given. Assumed as zero.")
    dipole_solvent = 0
if ((tot_dipole_solute == None) & (dipole_file == None)):
    print("Solute dipole not given. Assumed as zero.")
    dipole_solute = 0

    
# Thresholds/parameters and other useful constants
kT = (k*T)*(1.60218E-19) # kT in SI
u0 = c*nu0*h/(k*300) # Threshold for low frequency
                     # default = 100 cm^{-1} at 300 K
c_gas = (P/(8.314*T))/(1000) # M concentration of gas 

#########################
#### Useful functions ###
#########################
# xyz are the atomic coordinates
def centroid(x,y,z):
    C = [0,0,0]
    npts = len(x)
    for i in range(npts):
        C[0] = C[0] + x[i]
        C[1] = C[1] + y[i]         
        C[2] = C[2] + z[i]         
    C[0] = C[0]/npts; C[1] = C[1]/npts; C[2] = C[2]/npts
    return C

# xyz are the atomic coordinates
def center_mass(atoms,x,y,z):
    C = [0,0,0]
    npts = len(x)
    M_tot = 0.0
    for i in range(npts):
        mi = mass[atoms[i]]
        M_tot = M_tot + mi
        C[0] = C[0] + mi*x[i]
        C[1] = C[1] + mi*y[i]         
        C[2] = C[2] + mi*z[i]         
    C[0] = C[0]/M_tot; C[1] = C[1]/M_tot; C[2] = C[2]/M_tot
    return C
        
def gyradius(atoms,x,y,z):
    C = centroid(x,y,z)
    npts = len(atoms)
    Rg2 = 0
    for i in range(npts):
        dist = (C[0]-x[i])**2 + (C[1]-y[i])**2 + (C[2]-z[i])**2
        #dist = dist + r_vdW[atoms[i]]/100
        Rg2 = Rg2 + dist
    return m.sqrt(Rg2/npts)


# xyz are the atomic coordinates
def moment_inertia(atoms,x,y,z):
    Ixx = 0; Iyy = 0; Izz = 0
    Ixy = 0; Iyz = 0; Ixz = 0
    N_A = len(atoms)
    C = center_mass(atoms,x,y,z)
    I = 0.0
    for i in range(N_A):
        xi = (x[i]-C[0]); yi = (y[i]-C[1]); zi = (z[i]-C[2])
        Ixx = Ixx + mass[atoms[i]]*(yi**2 + zi**2)
        Iyy = Iyy + mass[atoms[i]]*(xi**2 + zi**2)
        Izz = Izz + mass[atoms[i]]*(xi**2 + yi**2)
        Ixy = Ixy - mass[atoms[i]]*(xi*yi)
        Iyz = Iyz - mass[atoms[i]]*(yi*zi)
        Ixz = Ixz - mass[atoms[i]]*(xi*zi)
    I_matrix = np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
    w = np.linalg.eig(I_matrix)[0]
    Ix,Iy,Iz = w[0],w[1],w[2]
    return Ix, Iy, Iz

# Maximum distance from centroid for each coordinate
# xyz are the atomic coordinates
def max_r(x,y,z):
    C = centroid(x,y,z)
    npts = len(x)
    x_max = 0; y_max = 0; z_max = 0
    for i in range(npts):
        xi = (x[i]-C[0])**2; yi = (y[i]-C[1])**2; zi = (z[i]-C[2])**2
        if (xi > x_max):
            x_max = xi
        if (yi > y_max):
            y_max = yi
        if (zi > z_max):
            z_max = zi
    return [m.sqrt(x_max),m.sqrt(y_max),m.sqrt(z_max)]

    

# Minimum bounding box surface area
def min_box_area(atoms,x,y,z):
    npts = len(x)
    x_max = x[0]; y_max = y[0]; z_max = z[0]
    x_min = x[0]; y_min = y[0]; z_min = z[0]
    for i in range(npts):
        xi = x[i] + r_vdW[atoms[i]]/100
        yi = y[i] + r_vdW[atoms[i]]/100 
        zi = z[i] + r_vdW[atoms[i]]/100
        #print(xi,yi,zi)
        if (xi > x_max):
            x_max = xi
        if (yi > y_max):
            y_max = yi
        if (zi > z_max):
            z_max = zi
        xi = x[i] - r_vdW[atoms[i]]/100
        yi = y[i] - r_vdW[atoms[i]]/100 
        zi = z[i] - r_vdW[atoms[i]]/100
        if (xi < x_min):
            x_min = xi
        if (yi < y_min):
            y_min = yi
        if (zi < z_min):
            z_min = zi
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min
    area = 2*(lx*ly + lx*lz + ly*lz)
    return area


# Minimum bounding box lengths
def min_box_lengths(atoms,x,y,z):
    npts = len(x)
    x_max = x[0]; y_max = y[0]; z_max = z[0]
    x_min = x[0]; y_min = y[0]; z_min = z[0]
    for i in range(npts):
        xi = x[i] + r_vdW[atoms[i]]/100
        yi = y[i] + r_vdW[atoms[i]]/100 
        zi = z[i] + r_vdW[atoms[i]]/100
        #print(xi,yi,zi)
        if (xi > x_max):
            x_max = xi
        if (yi > y_max):
            y_max = yi
        if (zi > z_max):
            z_max = zi
        xi = x[i] - r_vdW[atoms[i]]/100
        yi = y[i] - r_vdW[atoms[i]]/100 
        zi = z[i] - r_vdW[atoms[i]]/100
        if (xi < x_min):
            x_min = xi
        if (yi < y_min):
            y_min = yi
        if (zi < z_min):
            z_min = zi
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min
    return lx, ly, lz

def R_x(t,y,z):
    yr = []; zr = []
    for i in range(len(y)):
        yr.append(y[i]*m.cos(t) - z[i]*m.sin(t))
        zr.append(y[i]*m.sin(t) + z[i]*m.cos(t))
    return yr,zr

def R_y(t,x,z):
    xr = []; zr = []
    for i in range(len(x)):
        xr.append(x[i]*m.cos(t) + z[i]*m.sin(t))
        zr.append(-x[i]*m.sin(t) + z[i]*m.cos(t))
    return xr,zr

def R_z(t,x,y):
    xr = []; yr = []
    for i in range(len(y)):
        xr.append(x[i]*m.cos(t) - y[i]*m.sin(t))
        yr.append(x[i]*m.sin(t) + y[i]*m.cos(t))
    return xr,yr


# area of (truly) minimum bounding box
def min_bbox_area(atoms,x,y,z):
    npts = len(x)
    lx,ly,lz = min_box_lengths(atoms,x,y,z)
    vol = lx*ly*lz
    vol_max = vol
    area_max = 2*(lx*ly + lx*lz + ly*lz)
    nang = 36
    for i in range(nang):
        for j in range(nang):
            for k in range(nang):
                ti = i*(2*pi/nang)
                tj = j*(2*pi/nang)
                tk = k*(2*pi/nang)
                yr,zr = R_x(ti,y,z)
                xr,zr = R_y(tj,x,zr)
                xr,yr = R_z(tk,xr,yr)
                lxi,lyi,lzi = min_box_lengths(atoms,xr,yr,zr)
                voli = lxi*lyi*lzi
#                print(voli)
                if (voli < vol):
                    lx = lxi; ly = lyi; lz = lzi
                elif (voli > vol):
                    vol_max = voli
                    area_max = 2*(lxi*lyi + lxi*lzi + lyi*lzi)
    #print("vols = ", lx*ly*lz, vol_max)
    #print("area_max =", area_max)
    area = 2*(lx*ly + lx*lz + ly*lz)
    return area

# Maximum distance between two atoms for each coordinate
# including vdW radii
# xyz are the atomic coordinates
def max_dist(atoms,x,y,z):
    npts = len(atoms)
    x_max = 0; y_max = 0; z_max = 0
    for i in range(npts):
        for j in range(i,npts):
            xi = abs(x[i]-x[j]); yi = abs(y[i]-y[j]); zi = abs(z[i]-z[j])
            xi = xi + (r_vdW[atoms[i]] + r_vdW[atoms[j]])/100
            yi = yi + (r_vdW[atoms[i]] + r_vdW[atoms[j]])/100
            zi = zi + (r_vdW[atoms[i]] + r_vdW[atoms[j]])/100
            if (xi > x_max):
                x_max = xi
            if (yi > y_max):
                y_max = yi
            if (zi > z_max):
                z_max = zi
    return [x_max,y_max,z_max]
            

# Molecular volume evaluated with Monte Carlo
# xyz are the atomic coordinates
def mc_volume(atoms,x,y,z):
    N_A = len(x)
    x_box, y_box, z_box = max_r(x,y,z)
    x_box = x_box*6;  y_box = y_box*6; z_box = z_box*6
    # Replace zero lengths in case they are present by vdW radius
    max_r_vdW = 0.0
    for i in range(len(atoms)):
        ri = r_vdW[atoms[i]]/100.0
        if (ri > max_r_vdW):
            max_r_vdW = ri
        else:
            pass
    if (x_box < max_r_vdW):
        x_box = 4*max_r_vdW
    if (y_box < max_r_vdW):
        y_box = 4*max_r_vdW
    if (z_box < max_r_vdW):
        z_box = 4*max_r_vdW
    ##############################
    vol_box = x_box*y_box*z_box
    xc,yc,zc = centroid(x,y,z)
    npts_in = 0.0
    for i in range(npts_mc):
        xi = random.uniform(xc-x_box*0.5,xc+x_box*0.5)
        yi = random.uniform(yc-y_box*0.5,yc+y_box*0.5)
        zi = random.uniform(zc-z_box*0.5,zc+z_box*0.5)
        # Check if point is inside the molecule
        for j in range(N_A):
            dist = m.sqrt((xi-x[j])**2 + (yi-y[j])**2 + (zi-z[j])**2)
            if (dist < r_vdW[atoms[j]]/100.0):
                npts_in = npts_in + 1.0
                break
            else:
                pass
    vol_mol = vol_box*npts_in/npts_mc
    return vol_mol

# Molecular vdW surface area evaluated with Monte Carlo
# xyz are the atomic coordinates
def mc_area(atoms,x,y,z):
    N_A = len(x)
    dl = 0.1  # length of MC lines in Angstroms
    x_box, y_box, z_box = max_r(x,y,z)
    x_box = x_box*4;  y_box = y_box*4; z_box = z_box*4
    # Replace zero lengths in case they are present by covalent radius
    max_r_vdW = 0.0
    for i in range(len(atoms)):
        ri = r_vdW[atoms[i]]/100.0
        if (ri > max_r_vdW):
            max_r_vdW = ri
        else:
            pass
    if (x_box < max_r_vdW):
        x_box = 4*max_r_vdW
    if (y_box < max_r_vdW):
        y_box = 4*max_r_vdW
    if (z_box < max_r_vdW):
        z_box = 4*max_r_vdW
    ##############################
    vol_box = x_box*y_box*z_box
    xc,yc,zc = centroid(x,y,z)
    nlines_in = 0.0
    for i in range(npts_mc):
        xi = random.uniform(xc-x_box*0.5,xc+x_box*0.5)
        yi = random.uniform(yc-y_box*0.5,yc+y_box*0.5)
        zi = random.uniform(zc-z_box*0.5,zc+z_box*0.5)
        xf = random.uniform(-1,1)
        yf = random.uniform(-1,1)
        zf = random.uniform(-1,1)
        dli = m.sqrt(xf**2 + yf**2 + zf**2)
        xf1, yf1, zf1 = xi+xf*dl/dli, yi+yf*dl/dli, zi+zf*dl/dli
        xf2, yf2, zf2 = xi-xf*dl/dli, yi-yf*dl/dli, zi-zf*dl/dli
        # Check if only one of the line endpoints is inside the molecule
        for j in range(N_A):
            dist1 = m.sqrt((xf1-x[j])**2 + (yf1-y[j])**2 + (zf1-z[j])**2)
            dist2 = m.sqrt((xf2-x[j])**2 + (yf2-y[j])**2 + (zf2-z[j])**2)
            rj = r_vdW[atoms[j]]/100.0
            if ( ((dist1 < rj) & (dist2 > rj)) | ((dist1 > rj) & (dist2 < rj)) ):
                nlines_in = nlines_in + 1.0
                break
    area_mol = vol_box*nlines_in/(npts_mc*dl)
    return area_mol


start = time.time()
#############################
###### Compute Entropy ######
#############################
# Get the solute's number of atoms, moment of inertia, and geometric radius
N_line = 0
atoms_solute = []; x_solute = []; y_solute = []; z_solute = []
f_solute = open(solute_xyz,"r")
for line in f_solute.readlines():
    N_line = N_line + 1
    aline = line.split()    
    if (N_line == 1):
        NA_solute = int(aline[0])
    elif ((N_line >= 3) & (N_line < NA_solute+3)):
        aline[0] = aline[0].lower().title()        
        atoms_solute.append(aline[0])
        x_solute.append(float(aline[1]))
        y_solute.append(float(aline[2]))
        z_solute.append(float(aline[3]))
    else:
        pass
f_solute.close()

# Get the solvent's atoms and coordinates
N_line = 0
atoms_solvent = []; x_solvent = []; y_solvent = []; z_solvent = []
f_solvent = open(solvent_xyz,"r")
for line in f_solvent.readlines():
    N_line = N_line + 1
    aline = line.split()    
    if (N_line == 1):
        NA_solvent = int(aline[0])
    elif ((N_line >= 3) & (N_line < NA_solvent+3)):
        aline[0] = aline[0].lower().title()                
        atoms_solvent.append(aline[0])
        x_solvent.append(float(aline[1]))
        y_solvent.append(float(aline[2]))
        z_solvent.append(float(aline[3]))
    else:
        pass
f_solvent.close()

#
# Compute surface area and volumes of solute and solvent if not provided
# in the input file
if (area_solute == 0):
    area_solute = mc_area(atoms_solute,x_solute,y_solute,z_solute)
if (vol_solute == 0):
    vol_solute = mc_volume(atoms_solute,x_solute,y_solute,z_solute)
if (area_solvent == 0):
    area_solvent = mc_area(atoms_solvent,x_solvent,y_solvent,z_solvent)
if (vol_solvent == 0):
    vol_solvent = mc_volume(atoms_solvent,x_solvent,y_solvent,z_solvent)
m_solute = 0.0
for i in range(NA_solute):
    m_solute = m_solute + mass[atoms_solute[i]]
m_solvent = 0.0
for i in range(NA_solvent):
    m_solvent = m_solvent + mass[atoms_solvent[i]]
if (rho_solvent == None):
    print("Warning: The solvent density was not given. The volume of free")
    print("space per solvent particle will be assumed to be equal to the")
    print("volume of a solvent particle as these values are often similar.")
    vol_free = vol_solvent
    rho_solvent = (100**-3)*(m_solvent/N_Av)/(2*vol_solvent*1E-30)
else:
    # max possible density in g/mL
    rho_vdw = (100**-3)*(m_solvent/N_Av)/(vol_solvent*1E-30)  
    vol_free = vol_solvent*((rho_vdw/rho_solvent) - 1)
# Geometric factors 
if (mbb_solute == None):
    area_box = min_bbox_area(atoms_solute,x_solute,y_solute,z_solute)
else:
    area_box = mbb_solute
phi_solute = area_solute/area_box
print("Minimum Bounding Box Solute (Angstroms) = ", '%.4f'%area_box)

if (mbb_solvent == None):
    area_box = min_bbox_area(atoms_solvent,x_solvent,y_solvent,z_solvent)
else:
    area_box = mbb_solvent
phi_solvent = area_solvent/area_box
fac_phi = phi_solvent/phi_solute
print("Minimum Bounding Box Solvent (Angstroms) = ", '%.4f'%area_box)
print("Phi Solute = ", '%.4f'%phi_solute)    
print("Phi Solvent = ", '%.4f'%phi_solvent)

# Cavity radius in Angstroms
rc = (vol_solute**(1/3) + (vol_free**(1/3)))*((3/(4*pi))**(1/3))

### Alpha is Assumed zero if not given ####
if (alpha == None):
    print("Coefficient of thermal expansion alpha not given.")
    print("Assumed alpha = 0")
    alpha = 0 

    
###########################################################    
###### Compute Translational Contribution to Entropy ######
###########################################################
h2 = (6.626E-34)**2 # h^2 in SI units
msi = m_solute*(1.66054E-27) # mass in SI units
vc = ((vol_solute**(1/3) + (vol_free**(1/3)))**3)*((1E-10)**3) # vc in SI units
x = max(vol_free**(2/3) - vol_solute**(2/3),0)/(vol_solvent**(2/3) + vol_free**(2/3))
Nx = 4*((4*pi/3)**(2/3))*(rc**2)/(vol_free**(2/3) + vol_solvent**(2/3))
Nc = 1 + Nx*((1/(1-x)) - 1)
vc = vc*Nc
qt = ((2*pi*msi*kT/h2)**(1.5))*vc
S_t = k*(m.log(qt) + 1 + 1.5 )
# Gas phase translational entropy
qt_gas = ((2*pi*msi*kT/h2)**(1.5))*kT/P
S_t_gas = k*(m.log(qt_gas) + 1 + 1.5 )
#########################################################
print("Nc = ", '%.2f'%Nc)


########################################################    
###### Compute Rotational Contribution to Entropy ######
########################################################
    
# Average moment of inertia of solute
Ix,Iy,Iz = moment_inertia(atoms_solute,x_solute,y_solute,z_solute)
if (Ix < 10*np.finfo(float).eps):
    B_av = (Iy*Iz)**(1/2)
elif (Iy < 10*np.finfo(float).eps):
    B_av = (Ix*Iz)**(1/2)
elif (Iz < 10*np.finfo(float).eps):
    B_av = (Ix*Iy)**(1/2)
else:
    B_av = (Ix*Iy*Iz)**(1/3)
if (B_av == 0.):
    S_r = 0; S_r_gas = 0; S_v = 0; n_rot = 0
else:
    if (linear_mol):
        n_rot = 2
    else:
        n_rot = 3    
    B_av = B_av*((1E-10)**2)*(1.66054E-27)/(1.6022E-19) # unit conversion
    qr = m.sqrt((8*(pi**2)*k*T*B_av)/(h**2))
    S_r_gas = n_rot*k*(0.5 + m.log(qr) - m.log(sigma_r)/n_rot)
    if (n_rot == 3):
        S_r_gas = S_r_gas + k*0.5*m.log(pi)
    # qt approach
    Rg = gyradius(atoms_solute,x_solute,y_solute,z_solute)
    print("Gyradius Solute (Angstroms) = ", '%.4f'%Rg)    
    rcX = rc - Rg
    r_free = (3*vol_free/(4*pi))**(1/3)
    if (rcX < r_free):
        print("WARNINGX: rcX < r_free")
    vcX = ((4*pi/3)*rcX**3)*((1E-10)**3); vcX = vcX*Nc
    qtX = ((2*pi*msi*kT/h2)**(1.5))*vcX
    StX = k*(m.log(qtX) + 1 + 1.5 )
    S_r = S_r_gas + StX - S_t  
# SPT rotational entropy
if (n_rot > 0):
    S_r_spt = S_r 
else:
    S_r_spt = 0

# Rotational entropy for solvent
if (solute_xyz == solvent_xyz):
    S_r_gas2 = S_r_gas; S_r_spt2 = S_r_spt
else:
    Ix, Iy, Iz = moment_inertia(atoms_solvent,x_solvent,y_solvent,z_solvent)
    if (Ix < 10*np.finfo(float).eps):
        B_av2 = (Iy*Iz)**(1/2)
        n_rot2 = 2
    elif (Iy < 10*np.finfo(float).eps):
        B_av2 = (Ix*Iz)**(1/2)
        n_rot2 = 2
    elif (Iz < 10*np.finfo(float).eps):
        B_av2 = (Ix*Iy)**(1/2)
        n_rot2 = 2
    else:
        B_av2 = (Ix*Iy*Iz)**(1/3)
        n_rot2 = 3
    if (B_av2 == 0.):
        S_r_gas2 = 0; S_r_spt2 = 0
    else:
        B_av2 = B_av2*((1E-10)**2)*(1.66054E-27)/(1.6022E-19) # unit conversion
        qr = m.sqrt((8*(pi**2)*k*T*B_av2)/(h**2))
        S_r_gas2 = n_rot2*k*(0.5 + m.log(qr) - m.log(sigma_r2)/n_rot2)
        if (n_rot2 == 3):
            S_r_gas2 = S_r_gas2 + k*0.5*m.log(pi)
        rc2 = (vol_solvent**(1/3) +  vol_free**(1/3))*((3/(4*pi))**(1/3))
        # qt approach
        msi2 = m_solvent*(1.66054E-27)
        vc2 = ((4*pi/3)*rc2**3)*((1E-10)**3)
        x2 = max(vol_free**(2/3) - vol_solvent**(2/3),0)/(vol_solvent**(2/3) + vol_free**(2/3))
        Nx2 = 4*((4*pi/3)**(2/3))*(rc2**2)/(vol_free**(2/3) + vol_solvent**(2/3))
        Nc2 = 1 + Nx2*((1/(1-x2)) - 1)
        vc2 = vc2*Nc2
        qt2 = ((2*pi*msi2*kT/h2)**(1.5))*vc2        
        S_t2 = k*(m.log(qt2) + 1 + 1.5 )        
        Rg2 = gyradius(atoms_solvent,x_solvent,y_solvent,z_solvent)
        print("Gyradius Solvent (Angstroms) = ", '%.4f'%Rg2)    
        rc2X = rc2 - Rg2
        vc2X = ((4*pi/3)*rc2X**3)*((1E-10)**3)
        vc2X = vc2X*Nc2
        qt2X = ((2*pi*msi2*kT/h2)**(1.5))*vc2X
        S_t2X = k*(m.log(qt2X) + 1 + 1.5 )
        S_r_spt2 = S_r_gas2 +S_t2X - S_t2
        

DS_r0 =  S_r_spt2 - S_r_gas2
                
#######################################################



###########################################
##### Cavity Contributions to Entropy #####
###########################################
V_m = (1E+30)*(0.01**3)*(m_solvent/rho_solvent)/N_Av # A/molecule
rho_m = 1/V_m # number density
print("Solvent Number Density (molecules/Angstrom) = ", '%.3f'%rho_m)
# Free energy of cavity formation from Scaled Particle Theory
def SPT_G_cav(y,R):
    G_cav = k*T*(-m.log(1-y)   + R*3*y/(1-y)  + \
                 (3*y/(1-y) + (9/2)*((y/(1-y))**2))*((R)**2))
    return G_cav
# SPT cavity entropy
def SPT_S_cav(RM,RS,alpha_V):
    if (epsilon_r == 1):
        y = (pi/6)*((2*RS)**3)*rho_m; R = RM/RS
    else:
        y = (3/(4*pi))*(epsilon_r - 1)/(epsilon_r + 2); R = RM/RS    
    df_dy = -(-3*R*(-1+y) + (-1+y)**2 +(R**2)*(3+6*y))/((-1+y)**3)
    dy_dT = -alpha_V*y    
    dG_dT = SPT_G_cav(y,R)/T + k*T*df_dy*dy_dT 
    return -dG_dT

# S_w cavity entropy
if (omega != None):
    S_cav0 = -5.365*omega*k 
    S_ext = S_cav0 - DS_r0
    V_m = (1E+30)*(0.01**3)*(m_solvent/rho_solvent)/N_Av # A/molecule
    rho_m = 1/V_m # number density
    y = vol_solvent*rho_m # V_M*n_S in the paper
    S_cav = S_ext*(area_solute/area_solvent)*fac_phi + \
            k*(vol_solute/vol_solvent)*m.log(1-y) 
    R_M = ((3/(4*pi))**(1/3))*vol_solvent**(1/3)
    R_S = ((3/(4*pi))**(1/3))*vol_solvent**(1/3)
if (alpha != None):
    print("alpha(user)/K = ", alpha)
    R_M = ((3/(4*pi))**(1/3))*vol_solute**(1/3)
    R_S = ((3/(4*pi))**(1/3))*vol_solvent**(1/3)
    S_spt = SPT_S_cav(R_M,R_S,alpha)
    S_spt0 = SPT_S_cav(R_M,R_S,0)
    R_M = ((3/(4*pi))**(1/3))*vol_solvent**(1/3)
    R_S = ((3/(4*pi))**(1/3))*vol_solvent**(1/3)
    S_spt2 = SPT_S_cav(R_M,R_S,alpha)
    Px = -(kT/(vol_solvent*1E-30))*m.log(1-y)
    PLX2 = -vol_solvent*Px*1E-30/kT
    omega_SPT = (S_spt2 + S_r_spt2 - S_r_gas2 - k*PLX2)/(-5.365*k)
    omega_SPT2 = (SPT_S_cav(R_M,R_S,0) + \
                  S_r_spt2 - S_r_gas2 - k*PLX2)/(-5.365*k)
    if (omega == None):
        S_cav = -5.365*omega_SPT*k*(area_solute/area_solvent)*fac_phi
        omega = omega_SPT
    else:
        print("omega(user) = ", omega)
else:
    alpha = alpha_omega


################################################
##### Vibrational Contributions to Entropy #####
################################################
# Grimme's approach (Chem. Eur. J. 2012, 18, 9955) is employed

# Read frequencies
vib_count = 0
nu = []
if (linear_mol):
    N_vib = 3*NA_solute - 5
else:
    N_vib = 3*NA_solute - 6

if (freq_file == None):
    print("Warning: No frequencies file given. S_v will be assumed to be zero.")
else:
	f_freq = open(freq_file,"r")
	for line in f_freq.readlines():
	    aline = line.split()
	    if (len(aline) >= 3):
	        if (aline[0] == "Frequencies"):
	            vib_count = vib_count + 3
	            if (vib_count > N_vib):
	                nu = []
	                vib_count = 3 
	            nu.append(float(aline[2]))
	            if (len(aline) >= 4):
	                nu.append(float(aline[3])*freq_scal)
	            if (len(aline) >= 5):                
	                nu.append(float(aline[4])*freq_scal)
	    else:
	        pass
	f_freq.close()    

S_vib = 0; H_vib = 0
for i in range(0,len(nu)):
    if (nu[i] > 0.0):
        nui = nu[i]*c
        u = h*nui/(k*T)
        fac = m.exp(u) - 1
        H_vib = H_vib + h*nui*(0.5 + 1/fac)
        # Grimme's correction
        mu1 = h/(8*(pi**2)*nui)
        mu2 = mu1*B_av/(mu1 + B_av)
        x = m.sqrt((8*(pi**3)*k*T*mu2)/(h**2))
        S_r2 = k*(0.5 + m.log(x))
        S_v = k*(u/fac - m.log(1-m.exp(-u)))
        w = 1/(1+(u0/u)**4)
        S_vib = S_vib + w*S_v + (1-w)*S_r2

        
# Convert to cal/mol-K and compute total entropies        
S_r = S_r*23.06*1000; S_t = S_t*23.06*1000
S_vib = S_vib*23.06*1000; S_cav = S_cav*23.06*1000;
S_spt = S_spt*23.06*1000; S_spt0 = S_spt0*23.06*1000
S_r_gas = S_r_gas*23.06*1000; S_t_gas = S_t_gas*23.06*1000
S_omega = S_r + S_t + S_cav + S_vib
S_epsilon = S_r + S_t + S_spt0 + S_vib
S_alpha = S_r + S_t + S_spt + S_vib
S_gas = S_r_gas + S_t_gas + S_vib
DS_conc = -k*m.log(concentration/c_gas)*23.06*1000
DS_omega = S_omega - S_gas + DS_conc
DS_epsilon = S_epsilon - S_gas + DS_conc
DS_alpha = S_alpha - S_gas + DS_conc
H_sol = 0 ; H_vib = H_vib*23.06
H_t = k*T*(1.5 + P*vc/kT)*23.06
H_r = k*T*(n_rot/2)*23.06
H_tot = H_vib + H_t + H_r
end = time.time()
# Print results
print("Calculation time (s) = ", '%.3f'%(end - start))
print(" ------ Areas and Volumes in Angstrom ------ ")
print("Free volume per solvent particle  = ", '%.5f'%vol_free)
print("area_solute  = ", '%.5f'%area_solute)
print("vol_solute = ",  '%.5f'%vol_solute)
print("area_solvent = ",  '%.5f'%area_solvent)
print("vol_solvent = ",  '%.5f'%vol_solvent)
print(" ------ Enthalpies in kcal/mol ------ ")
print("H t r v sol = ", '%.2f'%H_t, '%.2f'%H_r, '%.2f'%H_vib,'%.2f'%H_sol)
print(" ------ Entropies in cal/mol-K ------ ") 
print("S_gas t r v = ",  '%.2f'%S_t_gas, '%.2f'%S_r_gas, '%.2f'%S_vib)
print("S_gas (Total) = ", '%.2f'%S_gas)
print("S_sol t r v cav (omega) = ", '%.2f'%S_t, '%.2f'%S_r, \
      '%.2f'%S_vib, '%.2f'%S_cav)
print("S_sol (Total-omega) = ", '%.2f'%S_omega)
print("S_sol t r v cav (epsilon) = ", '%.2f'%S_t, '%.2f'%S_r, \
      '%.2f'%S_vib, '%.2f'%S_spt0)
print("S_sol (epsilon) = ", '%.2f'%S_epsilon)
print("S_sol t r v cav (eps-alpha) = ", '%.2f'%S_t, '%.2f'%S_r, \
      '%.2f'%S_vib, '%.2f'%S_spt)
print("S_sol (eps-alpha) = ", '%.2f'%S_alpha)
print("R*ln(c_gas/c_sol) = ",  '%.2f'%DS_conc)
print("Solvation Entropies: ")
print("Omega Epsilon Eps-Alpha")
print('%.2f'%DS_omega, '%.2f'%DS_epsilon, '%.2f'%DS_alpha)
# Test
#print('%.2f'%S_t,'%.2f'%S_r,'%.2f'%S_vib,'%.2f'%S_cav,'%.2f'%S_spt0,'%.2f'%S_spt)
