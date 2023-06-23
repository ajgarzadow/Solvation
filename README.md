# Solvation

Calculation of solvation entropy according to 

J. Chem. Theory Comput. 2019, 15, 5, 3204–3214

If you utilize this code in please cite the above reference. 

The input file must define the following
mandatory arguments:
 solute = (xyz file for solute molecule)
 solvent = (xyz file for solvent molecule)
 omega = (Pitzer acentric factor of solvent)
 rho_solvent = (density of solvent in g/mL)

The following arguments are optional:
 freq_file = (a file with the vibrational frequencies in Gaussian format)
 linear = (True/False; whether the molecule is linear; default = False )
 symmetry = (rotational symmetry number of solute; default = 1)
 vol_solute = (Solute volume in A^3)
 area_solute = (Surface area of solute in A^2)
 vol_solvent = (Solvent volume in A^3)
 area_solvent = (Surface area of solvent in A^2)
 dipole_solute = (dipole moment of solute in Debye)
 dipole_solvent = (dipole moment of solvent in Debye)
 temperature = (temperature in Kelvin; default = 298.15)
 concentration = (concentration in mol/L of solute; default = 1M)
 nu0 = (low frequency threshold in cm^-1; default = 100 cm^-1)

Notes:
 - The format requires each variable to be specified in a separate
   line in the input file.
 - Providing the solute volume and surface area can save substantial
   computation time.
 - A list of Pitzer acentric factors for common solvents is
   available at:
http://www.kaylaiacovino.com/Petrology_Tools/Critical_Constants_and_Acentric_Factors.htm.
