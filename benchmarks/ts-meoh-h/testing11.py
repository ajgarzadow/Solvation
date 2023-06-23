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
from scipy.optimize import fsolve, bisect

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
mass = {"H" : 1.008, \
         "C" : 12.011, "N" : 14.007, "O" : 15.999, "Ne" : 20.18, \
         "P" : 30.974, "Cl" : 35.453, "Ar" : 39.948, \
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
r_vdW = {"H" : 120, \
         "C" : 170, "N" : 155, "O" : 152, "Ne" : 154, \
         "P" : 180, "Cl" : 175, "Ar" : 188, \
         "Ni" : 163, "I" : 198, "Kr" : 202, \
         "Xe" : 216}


# Default values for parameters/thresholds
T = 298.15
P = 100000.0 # Pressure in Pa 
concentration = 1 # M
sigma_r = 1 # rotational symmetry number
epsilon_r = 78.54 # Relative permittivity (default is water)
sigma_0 = 72.8E-3 # Surface tension coefficient (default is water)
dsigma_dT = -0.1514E-3 # Temperature coefficient (default is water)
alpha = 207E-6  # Isobaric thermal volumetric expansion coefficient (default is water)
nu0 = 100 # Threshold for low frequency in Grimme's hindered rotor treatment of vibrations (in cm^-1)
nu0_r = 100 # Same as nu0 but for treating vibrations that correspond to rotations in a solvent cage
nu_cage = 20 # Frequency of vibrations associated with rotational motions of the solute in a solvent cage
npts_mc = 1000000 # Number of points for Monte Carlo molecular volume evaluation
omega = None # Pitzer acentric factor
linear_mol = False
solute_xyz = None
solvent_xyz = None
freq_file = None

# Initialize other optional input variables
rho_solvent = None # density of solvent g/mL
area_solute = 0; vol_solute = 0
area_solvent = 0; vol_solvent = 0
dipole_solvent = None; dipole_solute = None

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
            dipole_solute = float(aline[2])            
        elif (aline[0]  == 'dipole_solvent'):            
            dipole_solvent = float(aline[2])
        elif (aline[0]  == 'rho_solvent'):            
            rho_solvent = float(aline[2])                                    
        elif (aline[0]  == 'epsilon_r'):            
            epsilon_r = float(aline[2])                                                      
f_directives.close()

if (omega == None):
    print("Error: The Pitzer acentric factor omega must be defined.")
    sys.exit()
elif (solute_xyz == None):
    print("Error: The solute xyz file name must be given.")
    sys.exit()
elif (solvent_xyz == None):
    print("Error: The solvent xyz file name must be given.")
    sys.exit()

if (dipole_solvent == None):
    print("Solvent dipole not given. Assumed as zero")
    dipole_solvent = 0
if (dipole_solute == None):
    print("Solute dipole not given. Assumed as zero")
    dipole_solute = 0
    
# Thresholds/parameters and other useful constants
kT = (k*T)*(1.60218E-19) # kT in SI
u0 = c*nu0*h/(k*300) # Threshold for low frequency
                     # default = 100 cm^{-1} at 300 K
u0_r = c*nu0_r*h/(k*300) 
nu_cage = nu_cage*c
u = h*nu_cage/(k*T)

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
    if (Ix < 10*np.finfo(float).eps):
        I = (Iy*Iz)**(1/2)
    elif (Iy < 10*np.finfo(float).eps):
        I = (Ix*Iz)**(1/2)
    elif (Iz < 10*np.finfo(float).eps):
        I = (Ix*Iy)**(1/2)
    else:
        I = (Ix*Iy*Iz)**(1/3)
    return I

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
    x_box = x_box*4;  y_box = y_box*4; z_box = z_box*4
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
    rho_vdw = (100**-3)*(m_solvent/N_Av)/(vol_solvent*1E-30)
    vol_free = vol_solvent*((rho_vdw/rho_solvent) - 1)
rc = (vol_solute**(1/3) + (vol_free**(1/3)))*((3/(4*pi))**(1/3))    

area_box = min_box_area(atoms_solute,x_solute,y_solute,z_solute)
phi_solute = area_solute/area_box
area_box = min_box_area(atoms_solvent,x_solvent,y_solvent,z_solvent)
phi_solvent = area_solvent/area_box
fac_phi = phi_solvent/phi_solute 

########################################################    
###### Compute Rotational Contribution to Entropy ######
########################################################    
# Average moment of inertia of solute
B_av = moment_inertia(atoms_solute,x_solute,y_solute,z_solute)
if (B_av == 0.):
    S_r = 0; S_r_gas = 0; S_v = 0
    S_ri0 = 0; S_rif = 0; S_ri1 = 0; S_ri2 = 0; S_ri3 = 0; S_ri4 = 0
    S_r_mech = 0; S_r_mu = 0; n_rot = 0
else:
    if (linear_mol):
        n_rot = 2
    else:
        n_rot = 3    
    B_av = B_av*((1E-10)**2)*(1.66054E-27)/(1.6022E-19) # unit conversion
    mu1 = h/(8*(pi**2)*nu_cage)
    mu2 = mu1*B_av/(mu1 + B_av)
    fac = m.exp(u) - 1
    qr_mu = m.sqrt((8*(pi**3)*k*T*mu2)/(h**2))
    qr = m.sqrt((8*(pi**3)*k*T*B_av)/(h**2))
    w = 1/(1+(u0_r/u)**4)
    S_v = k*(u/fac - m.log(1-m.exp(-u)))
    S_r = k*(0.5 + m.log(qr) - m.log(sigma_r)/n_rot)
    S_r_gas = n_rot*k*(0.5 + m.log(qr) - m.log(sigma_r)/n_rot)
    S_r_mu = w*S_v + (1-w)*k*(0.5 + m.log(qr_mu)) #- (1-w)*k*m.log(sigma_r)/n_rot
    rab = max((area_solute - area_solvent)/(area_solvent),0)
    if (abs(dipole_solute - dipole_solvent) < dipole_solute):
        dipole_eff = abs(dipole_solute - dipole_solvent)
    else:
        dipole_eff = dipole_solute
    kirkwood_f = 2*(epsilon_r - 1)/(2*epsilon_r + 1)
    dipole_int = kirkwood_f*1E-30*dipole_eff*dipole_solvent*(3.335**2)/ \
                 (4*pi*epsilon_0*(rc**3))
    rab = rab*fac_phi
    m_int = 2/pi
    p_rot0 = m.exp(-rab/n_rot) * \
            m.exp(-m_int*dipole_int/kT)
    p_not = 1-p_rot0
    p_rot = sigma_r*p_rot0/(p_not+sigma_r*p_rot0)
    S_r = S_r_gas*p_rot + n_rot*S_r_mu*(1 - p_rot)
    p_rot0 = m.exp(-rab/n_rot)
    p_not = 1-p_rot0
    p_rot = sigma_r*p_rot0/(p_not+sigma_r*p_rot0)
    S_r_mech = S_r_gas*p_rot + n_rot*S_r_mu*(1 - p_rot)
# SPT rotational entropy
if (n_rot > 0):
    rab = (area_solute/area_solvent)*fac_phi
    dipole_int = kirkwood_f*1E-30*dipole_solute*dipole_solvent*(3.335**2)/ \
                 (4*pi*epsilon_0*(rc**3))
    m_int = 2/pi
    p_rot0 = m.exp(-rab/n_rot) * \
             m.exp(-m_int*dipole_int/kT)
    p_not = 1-p_rot0
    p_rot = sigma_r*p_rot0/(p_not+sigma_r*p_rot0)
    S_r_spt = S_r_gas*p_rot + n_rot*S_r_mu*(1 - p_rot)
else:
    S_r_spt = 0



    
#######################################################

###########################################################    
###### Compute Translational Contribution to Entropy ######
###########################################################
h2 = (6.626E-34)**2
msi = m_solute*(1.66054E-27)
vc = ((4*pi/3)*rc**3)*((1E-10)**3)#*m.sqrt(3)*pi/2#4*pi/3
#vc = vol_solute*((1E-10)**3)*m.sqrt(3)*pi/2#4*pi/3
qt_gas = ((2*pi*msi*kT/h2)**(1.5))*kT/P
qt = ((2*pi*msi*kT/h2)**(1.5))*vc*np.exp(-P*vc/kT)
c_gas = (P/(8.314*T))/(1000) # M concentration of gas (to be used later)
S_t = k*(m.log(qt) + 1 + 1.5 + P*vc/kT)
S_t_gas = k*(m.log(qt_gas) + 1 + 1.5 )
#########################################################


###########################################
##### Cavity Contributions to Entropy #####
###########################################
S_cav = -5.365*omega*k*(area_solute/area_solvent)*fac_phi

# Quick enthalpy estimate (water, SI units)
k_h = (S_cav*23.06*4186/N_Av)/ \
      ((dsigma_dT + 1.5*alpha*sigma_0)*area_solute*(1E-20)) # SI units
A_0 = 1 - T*(dsigma_dT/sigma_0 + 1.5*alpha)
DH_mech = N_Av*sigma_0*A_0*area_solute*(1E-20)*k_h
Delta_H0 = 5.365*(1+omega)*k*674*(23.06)*4186 # SI units
S_cav0 = -5.365*omega*k
k_h0 = (S_cav0*23.06*4186/N_Av)/ \
       ((dsigma_dT + 1.5*alpha*sigma_0)*area_solvent*(1E-20))# SI units
DH_mech0 = N_Av*sigma_0*A_0*area_solvent*(1E-20)*k_h0
rc0 = (vol_solvent**(1/3) + vol_free**(1/3))*((3/(4*pi))**(1/3))
kirkwood_f = 2*(epsilon_r - 1)/(2*epsilon_r + 1)
dipole_int0 = 0.5*kirkwood_f*1E-30*dipole_solvent*dipole_solvent*(3.335**2)/ \
             (4*pi*epsilon_0*(rc0**3))
if (dipole_int0 > 0 ):
    N_kiss0 = (Delta_H0 - DH_mech0)/(N_Av*dipole_int0*(8/(4*pi)))
else:
    N_kiss0 = 0
dipole_int = 0.5*kirkwood_f*1E-30*dipole_solute*dipole_solvent*(3.335**2)/ \
             (4*pi*epsilon_0*(rc**3))
Delta_H = DH_mech + N_kiss0*N_Av*dipole_int*(8/(4*pi))*(area_solute/area_solvent)*fac_phi
Delta_H = -Delta_H/4186


# Scaled particle theory enthalpy estimate
V_m = (1E+30)*(0.01**3)*(m_solvent/rho_solvent)/N_Av # A/molecule
rho_m = 1/V_m
drho_dT = -rho_m*alpha 
R_M = ((3/(4*pi))**(1/3))*vol_solute**(1/3)
R_S = rc - R_M
R0 = R_M/R_S
y0 =(pi/6)*((2*R_S)**3)*rho_m
# Free energy of cavity formation from Scaled Particle Theory
def SPT_G_cav(y,R):
    G_cav = k*T*(-m.log(1-y) + (R)*3*y/(1-y) + \
                 (3*y/(1-y) + (9/2)*((y/(1-y))**2))*((R)**2))
    return G_cav
# SPT cavity entropy
def SPT_S_cav(y,R):
    df_dy = -((R**2)*(-6*y+3)+3*R*(y-1)-((y-1)**2))/((1-y)**3)
    dy_dT = (y/rho_m)*drho_dT
    dG_dT = (SPT_G_cav(y,R)/T)  + \
            k*T*df_dy*dy_dT
    return -dG_dT
def equations(RMS):
    R_M, R_S = RMS
    R = R_M/R_S
    y = (pi/6)*((2*R_S)**2)*rho_m
    vol_ratio = (vol_solvent**(1/3))/(vol_solute**(1/3))
    eqn0 = SPT_S_cav(y,R*vol_ratio) - S_cav0
    eqn1 = SPT_S_cav(y,R) - S_cav
    return (eqn0, eqn1)
def equations2(y):    
    eqn1 = SPT_S_cav(y,R0) - S_cav
    return eqn1
def equations3(R):    
    eqn1 = SPT_S_cav(y0,R) - S_cav
    return eqn1
def equations4(R_M):
    R_S = rc - R_M
    y0 = (pi/6)*((2*R_S)**2)*rho_m
    R = R_M/R_S
    eqn1 = -SPT_S_cav(y0,R) - S_cav
    return eqn1


#R_M, R_S = fsolve(equations, (R_M,R_S))
#y = fsolve(equations2, y0); R = R0
#R = fsolve(equations3, R0); y = y0
#R_M = fsolve(equations4, R_M)[0]
#R_M = bisect(equations4, 1, rc-1, xtol=1E-5)
R_S = rc - R_M
y = (pi/6)*((2*R_S)**2)*rho_m
R = R_M/R_S
S_spt = SPT_S_cav(y,R)*23.06*1000
print("SPT S_cav = ", S_spt)
#DS_conc = -k*m.log(concentration/c_gas)*23.06
print("SPT G_cav = ", SPT_G_cav(y,R)*23.06)
#S_spt = S_cav*23.06*1000
DH_SPT = SPT_G_cav(y,R)*23.06 + T*(S_spt)/1000 - (1/4186)*\
         N_kiss0*N_Av*dipole_int*(8/(4*pi))*(area_solute/area_solvent)*fac_phi
print("DH_mech 12 = ", DH_mech/4186, SPT_G_cav(y,R)*23.06 + T*(S_spt)/1000)



#print(y)
#G_cav = k*T*(-m.log(1-y) + (R_M/R_S)*3*y/(1-y) + \
#             (3*y/(1-y) + (9/2)*((y/(1-y))**2))*((R_M/R_S)**2))
#Delta_H = -(G_cav + T*S_cav)*23.06

#Delta_H = -5.365*(1+omega)*k*674*23.06*(area_solute/area_solvent)*fac_phi
#fac_vol = vol_solvent/vol_solute
#a_r = (8/(4*pi))*fac_phi*(area_solute/area_solvent)*fac_vol
#fac = (dipole_solute + a_r*dipole_solvent)/((1+a_r)*dipole_solvent)
#Delta_H = Delta_H*fac

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
	                nu.append(float(aline[3]))
	            if (len(aline) >= 5):                
	                nu.append(float(aline[4]))
	    else:
	        pass
	f_freq.close()    

S_vib = 0
for i in range(0,len(nu)):
    if (nu[i] > 0.0):
        nui = nu[i]*c
        u = h*nui/(k*T)
        fac = m.exp(u) - 1        
        # Grimme's correction
        mu1 = h/(8*(pi**2)*nui)
        mu2 = mu1*B_av/(mu1 + B_av)
        x = m.sqrt((8*(pi**3)*k*T*mu2)/(h**2))
        S_r2 = k*(0.5 + m.log(x))
        S_v = k*(u/fac - m.log(1-m.exp(-u)))
        w = 1/(1+(u0/u)**4)
        S_vib = S_vib + w*S_v + (1-w)*S_r2


# Concer to cal/mol-K and compute total entropies        
S_r = S_r*23.06*1000; S_t = S_t*23.06*1000
S_vib = S_vib*23.06*1000; S_cav = S_cav*23.06*1000
S_r_mech = S_r_mech*23.06*1000; S_r_spt = S_r_spt*23.06*1000
S_r_gas = S_r_gas*23.06*1000; S_t_gas = S_t_gas*23.06*1000
S_tot = S_r + S_t + S_cav + S_vib
S_tot_mech = S_r_mech + S_t + S_cav + S_vib
S_tot_free = S_r_gas + S_t + S_cav + S_vib
S_tot_spt = S_r_spt + S_t + S_spt + S_vib
S_gas = S_r_gas + S_t_gas + S_vib
DS_conc = -k*m.log(concentration/c_gas)*23.06*1000
DS = S_tot - S_gas + DS_conc
DS_mech = S_tot_mech - S_gas + DS_conc
DS_free = S_tot_free - S_gas + DS_conc
DS_spt = S_tot_spt - S_gas + DS_conc
end = time.time()
# Print results
print("Calculation time (s) = ", '%.3f'%(end - start))
print("Solvent kissing No. = ", '%.2f'%N_kiss0)
print(" ------ Areas and Volumes in Angstrom ------ ")
print("Free volume per solvent particle  = ", '%.5f'%vol_free)
print("area_solute  = ", '%.5f'%area_solute)
print("vol_solute = ",  '%.5f'%vol_solute)
print("area_solvent = ",  '%.5f'%area_solvent)
print("vol_solvent = ",  '%.5f'%vol_solvent)
print(" ------ Entropies in cal/mol-K ------ ") 
print("S_gas t r v = ",  '%.2f'%S_t_gas, '%.2f'%S_r_gas, '%.2f'%S_vib)
print("S_gas (Total) = ", '%.2f'%S_gas)
print("S_sol t r v cav (FreeRot) = ", '%.2f'%S_t, '%.2f'%S_r_gas, \
      '%.2f'%S_vib, '%.2f'%S_cav)
print("S_sol (Total-FreeRot) = ", '%.2f'%S_tot_free)
print("S_sol t r v cav (Mech) = ", '%.2f'%S_t, '%.2f'%S_r_mech, \
      '%.2f'%S_vib, '%.2f'%S_cav)
print("S_sol (Total-Mech) = ", '%.2f'%S_tot_mech)
print("S_sol t r v cav (MechElec) = ", '%.2f'%S_t, '%.2f'%S_r, \
      '%.2f'%S_vib, '%.2f'%S_cav)
print("S_sol (Total-MechElec) = ", '%.2f'%S_tot)
print("R*ln(c_gas/c_sol) = ",  '%.2f'%DS_conc)
print("Solvation Entropies: ")
print("FreeRot Mech MechElec SPT")
print('%.2f'%DS_free,'%.2f'%DS_mech,'%.2f'%DS, '%.2f'%DS_spt)
print(" Delta H: ")
print('%.2f'%Delta_H, '%.2f'%DH_SPT)
#print(" Delta G: ")
#Delta_G = Delta_H - T*DS/1000
#print(' %.2f'%Delta_G)
