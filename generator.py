
##
# \file
# \brief Data generation script for proper orthogonal decomposition.
# \author Copyright (C) 2018-2020 Modesar SHAKOOR
# \version 3.0
# \date 7 February 2020
#
# This file is part of Fems.
#
# Fems is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# Fems is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with Fems. If not, see <http://www.gnu.org/licenses/>.
#

import math
import matplotlib.pyplot
import numpy
import os
import random
import subprocess
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

# read command line arguments
# EXAMPLES:
# python3 generator.py 6 11
# python3 generator.py 128 11
## Number of strain paths to generate and solve.
M = int(sys.argv[1])
## Number of time increments for each strain path.
NT = int(sys.argv[2])
## Final time for all simulations (time step is \ref T divided by (\ref NT - 1)).
T = 0.001

subprocess.call(["mkdir", "database"])

# randomly shoot a strain path and then interpolate it for smoothing
## Values of time at each time increment.
t = numpy.arange(0, 1+1/(NT-1)/2, 1/(NT-1))
## Randomly generated and smoothed strain paths.
StrainPoints = numpy.zeros([M,NT,3])
## Number of randomly generated strain paths control points.
NSCPs = 2
## Randomly generated strain paths control points.
StrainControlPoints = numpy.zeros([M,NSCPs,3])
for n in numpy.arange(1,NT):
	for i in numpy.arange(0,3):
		StrainPoints[i,n,i] = n*T/(NT-1);
		StrainPoints[3+i,n,i] = -n*T/(NT-1);
for m in numpy.arange(3*2,M):
	for n in numpy.arange(1,NSCPs):
		for i in numpy.arange(0,3):
			StrainControlPoints[m,n,i] = StrainControlPoints[m,n-1,i] + random.uniform(-T/NSCPs,T/NSCPs)
	tck, u = interpolate.splprep(StrainControlPoints[m,:,:].reshape([NSCPs,3]).transpose(), s=0)
	out = interpolate.splev(t,tck)
	for i in numpy.arange(0,3):
		StrainPoints[m,:,i] = out[i]

print(StrainPoints.shape)

# show all smoothed strain paths
fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
for m in numpy.arange(0,M):
	ax.plot(StrainControlPoints[m,:,0],StrainControlPoints[m,:,1],StrainControlPoints[m,:,2], 'x')
	ax.plot(StrainPoints[m,:,0],StrainPoints[m,:,1],StrainPoints[m,:,2], '>-')
matplotlib.pyplot.show()

# convert to 2*2 strain matrices
StrainPoints_new = numpy.zeros([M,NT,2,2])
StrainPoints_new[:,:,0,0] = StrainPoints[:,:,0]
StrainPoints_new[:,:,0,1] = StrainPoints[:,:,1]
StrainPoints_new[:,:,1,0] = StrainPoints[:,:,1]
StrainPoints_new[:,:,1,1] = StrainPoints[:,:,2]
StrainPoints = StrainPoints_new

# create database files and run FEMS
os.chdir("database")
if os.getcwd().endswith("database"): # this check is important to avoid removing files in the wrong directory
	subprocess.call(["rm *.npy"], shell=True)
	subprocess.call(["rm *.log"], shell=True)

	for m in numpy.arange(0,M):
		# write strain points in numpy format (time step by time step)
		numpy.save("StrainPoints_"+str(m), StrainPoints[m,:,:,:])
		subprocess.call(["python3", "../elastoplasticite.py", "./StrainPoints_"+str(m)+".npy"], stdout=open("run_"+str(m)+".log", "w"), stderr=subprocess.STDOUT)
		subprocess.call(["mv strain.npy strain_"+str(m)+".npy"], shell=True)
		subprocess.call(["mv stress.npy stress_"+str(m)+".npy"], shell=True)
		subprocess.call(["mv strain_pl_equiv.npy strain_pl_equiv_"+str(m)+".npy"], shell=True)
		subprocess.call(["mv strain_pl.npy strain_pl_"+str(m)+".npy"], shell=True)
os.chdir("..")
