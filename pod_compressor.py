
##
# \file
# \brief Proper Orthogonal Decomposition (POD) script.
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

import glob
import numpy
import re
import sys
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# read command line arguments
# EXAMPLES:
# python3 pod_compressor.py "database/strain_pl_[0-9]*.npy" 225 4 1 6 "strain_pl_"
## Path to the data.
data_path = sys.argv[1]
## Number of mesh nodes.
N = int(sys.argv[2])
## Space dimension.
D = int(sys.argv[3])
## Whether to convert or load a converted data.
convert = int(sys.argv[4]) == 1
## Number of components for the POD.
K = int(sys.argv[5])
## Prefix for output files.
prefix = "_"
if(len(sys.argv) > 6):
	prefix = sys.argv[6]

## Number of data files.
data_file_count = len(glob.glob(data_path))
## Whether to split the data into a training dataset and a testing dataset
# and then compute the compression error on the testing dataset
# when using the POD vectors computed using only the training dataset.
doTest = False

# for big datasets: use mmap_mode='r' as option in numpy.load

##
# \brief Function used for sorting data points in ascending order.
# \details The path to a data file may contain one or two integers.
# The files are sorted in ascending order with respect to these integers.
# For instance, the first integer may represent a loading path, and the second a load increment.
# \param data_file A data file path.
# \return A unique key used to sort data files.
#
def data_path_key(data_file):
	keys = re.findall("(\d+)", data_file)
	if len(keys) > 1:
		return int(keys[0])*data_file_count + int(keys[1])
	else:
		return int(keys[0])

## Training data, each row is a training data point.
data_train = None
## Testing data, each row is a testing data point.
data_test = None
if convert:
	if len(glob.glob(data_path)) == 0:
		print("No file matches path %s" % data_path)
		exit(0)
	## Data, each row is a data point.
	data = None
	for data_file in sorted(glob.glob(data_path), key=data_path_key):
		print("Loading data file %s" % data_file)
		data_file = numpy.load(data_file)
		## Number of data points in this data file.
		M = int(data_file.reshape(-1).shape[0] / N / D)
		# data_file = numpy.rollaxis(data_file.reshape([M, D, N]), 2, 1)
		data_file = data_file.reshape([M, N*D])
		print("Extracted shape %s" % str(data_file.shape))
		if data is None:
			data = data_file
		else:
			data = numpy.concatenate((data,data_file),axis=0)

	print("Data shape: %s" % str(data.shape))

	# split training and testing data
	if doTest:
		data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
	else:
		data_train = data
	data = None
	print("Training data shape: %s" % str(data_train.shape))
	if doTest:
		print("Testing data shape: %s" % str(data_test.shape))

	# dump training/testing database
	numpy.save(prefix+'data_train', data_train)
	if doTest:
		numpy.save(prefix+'data_test', data_test)
else:
	# read training/testing database
	data_train = numpy.load(prefix+'data_train.npy')
	if doTest:
		data_test = numpy.load(prefix+'data_test.npy')

## POD compression (principal component analysis is just one way to compute the POD).
compresser = PCA(n_components=K)
# compute POD vectors using only the training dataset
compresser = compresser.fit(data_train)
## Final number of POD vectors (might be less than user-defined one if not enough independent data).
K = compresser.components_.shape[0]
print("Number of POD vectors %s" % str(K))

## Compressed training data.
H_train = compresser.transform(data_train)
# compute and display approximation error on the training dataset
for k in numpy.arange(0,K):
	print("Compresser explained variance ratio by weight vector %ld in training %e" % (k, compresser.explained_variance_ratio_[k]))
print("Compresser total explained variance ratio in training %e" % numpy.sum(compresser.explained_variance_ratio_))
print("Compresser relative approximation error in training %e" % (numpy.linalg.norm(compresser.inverse_transform(H_train)-data_train) / numpy.linalg.norm(data_train)))

if doTest:
	## Compressed testing data, using POD vectors computed using only the training dataset.
	H_test = compresser.transform(data_test)
	# compute and display approximation error on the testing dataset
	print("Compresser relative approximation error in testing %e" % (numpy.linalg.norm(compresser.inverse_transform(H_test)-data_test) / numpy.linalg.norm(data_test)))

# dump compresser and principal components
joblib.dump(compresser, prefix+'compresser.pkl')
numpy.save(prefix+"vectors_"+str(K), compresser.components_.reshape(K,N,D))
