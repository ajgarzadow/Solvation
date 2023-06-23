#!/usr/bin/python

import sys
import os
import string
from math import*


scoord = sys.argv[1]
xyz = scoord +'.gjf'
f = open(scoord,"r")
fxyz = open(xyz,"w")
line_offset = []
offset = 0
count = 0
final = 0
nat = 0


for line in f.readlines():
    aline = line.split()
    if (aline[0] == '$end'):
        break
    elif (aline[0] == '$coord'):
        pass
    else:
            atom = str(aline[3])
            x = ' ' +str(aline[0])
            y = ' ' + str(aline[1])
            z = ' ' + str(aline[2])
            fxyz.write(atom + x + y + z + '\n' )
            nat = nat + 1            
fxyz.close()
f.close()

os.system('rm temp')
l1 = '%chk=calc.chk'
l2 = '%nprocshared=10'
l3 = '%mem=16GB'
l4 = '# opt=loose freq wb97xd/6-31g(d) 5d scf=(maxconventionalcycles=100,xqc) units=au'
tmp = open("temp","w")
tmp.write(l1 + ' \n')
tmp.write(l2 + ' \n')
tmp.write(l3 + ' \n')
tmp.write(l4 + ' \n')
tmp.close()
#os.system('echo ' + l1 + ' >> temp')
#os.system('echo ' + l2 + ' >> temp')
#os.system('echo ' + l3 + ' >> temp')
#os.system('echo ' + l4 + ' >> temp')
os.system('echo  >> temp')
os.system('echo comment >> temp')
os.system('echo  >> temp')
os.system('echo 0 1 >> temp')
os.system('cat ' + xyz + ' >> temp')
os.system('cp temp ' + xyz)
os.system('echo  >> ' + xyz)


