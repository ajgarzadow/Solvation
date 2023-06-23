#!/usr/bin/python

import sys
import os
import string
from math import*


gjf = sys.argv[1]
xyz = gjf[0:-4] +'.xyz'
f = open(gjf,"r")
fxyz = open(xyz,"w")
line_offset = []
offset = 0
count = 0
final = 0
nat = 0


count = 0
for line in f.readlines():
    count = count + 1
    aline = line.split()
    if (count >= 7):
        if (len(aline) == 4):
            atom = str(aline[0])
            x = ' ' +str(aline[1])
            y = ' ' + str(aline[2])
            z = ' ' + str(aline[3])
            fxyz.write(atom + x + y + z + '\n' )
            nat = nat + 1            
        else:            
            break
fxyz.close()
f.close()

os.system('echo ' + str(nat) +' > temp')
os.system('echo comment >> temp')
os.system('cat ' + xyz + ' >> temp')
os.system('cp temp ' + xyz)
os.system('echo  >> ' + xyz)




