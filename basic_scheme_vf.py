# -*-coding:utf-8 -*
import numpy as np
import scipy as sp
import math as m
import matplotlib.pyplot as plt

d = 2

# behaviour law for linear elastic isotropic material
# strain = scalaire en 1D, matrice 2x2 en 2D, matrice 3x3 en 3D
# cette fonction retourne stress = scalaire en 1D, matrice 2x2 en 2D, matrice 3x3 en 3D


#used functions
#stress calculation
def compute_behaviour(mu, Lambda, deformation):
	trace_deformation = 0
	for iid in range(0,d):
		trace_deformation = trace_deformation + deformation[iid,iid]
	return 2*mu*deformation +  Lambda*trace_deformation*np.eye(d)



# domain definition
Lx = 1       # domain size in x direction
N = np.zeros(d,dtype=np.int)      # number of voxels in x direction
N[0] = 15
N[1] = 15
mu = np.zeros(N)
Lambda = np.zeros(N)     #J'utilise lamba avec majuscule car celui avec
                          # minuscule est un mot reserve de python

for ix in range(0,N[0]):
	for iy in range(0,N[1]):
		if(ix>(N[0]/4) and iy>(N[1]/4) and ix<(N[0]*3/4) and iy<(N[1]*3/4)):
			mu[ix,iy] = 1.
		else:
			mu[ix,iy] = 10.
		Lambda[ix,iy] = 1.

# wisely choose reference material parameters
mu0 = 0.5*(mu.max() + mu.min())
Lambda0 = 0.5*(Lambda.max() + Lambda.min())

# boundary conditions
strain_macro = np.zeros((d,d))
strain_macro[0,0] = 0.01

# variables
strain = np.zeros(np.concatenate((N,d,d),axis=None))

#symbole de Kronecker
def delta(a,b):
	if (a==b):
		kronecker = 1.
	else:
		kronecker = 0.
	return kronecker
#Green's operator
tf_gamma0 = np.zeros(np.concatenate((N,d,d,d,d),axis=None))
for ix in range(0,N[0]):
	for iy in range(0,N[1]):
		ksi = np.zeros(d);
		if(N[0] % 2 == 0):
			ksi[0] = -(N[0]/2) + ix
		else:
			ksi[0] = -((N[0]-1)/2) + ix
		if(N[1] % 2 == 0):
			ksi[1] = -(N[1]/2) + iy
		else:
			ksi[1] = -((N[1]-1)/2) + iy
		ksi_2 = np.linalg.norm(ksi)**2
		ksi_4 = ksi_2**2
		if ksi_2 > 0:
			for iid in range(0,d):
				for jd in range(0,d):
					for kd in range(0,d):
						for hd in range(0,d):
							tf_gamma0[ix,iy,iid,jd,kd,hd] = (delta(kd,iid)*ksi[hd]*ksi[jd] + delta(hd,iid)*ksi[kd]*ksi[jd] + delta(kd,jd)*ksi[hd]*ksi[iid] + delta(hd,jd)*ksi[kd]*ksi[iid])/(4*mu0*ksi_2) - ((Lambda0+mu0)/(mu0*(Lambda0+2*mu0)))*(ksi[iid]*ksi[jd]*ksi[kd]*ksi[hd])/ksi_4;
#tf_gamma0 = np.fft.ifftshift(tf_gamma0, axes=range(0,d))

# besoin de .reshape((d,d)) ???
#contrainte à la première iteration
stress = np.zeros(np.concatenate((N,d,d),axis=None))
for ix in range(0,N[0]):
	for iy in range(0,N[1]):
		stress[ix,iy,:,:] = compute_behaviour(mu[ix,iy], Lambda[ix,iy], strain[ix,iy,:,:])

# algorithm
residual = 1
iteration = 0
while residual > 1e-4:

	#compute polarization stress
	tau = np.zeros(np.concatenate((N,d,d),axis=None))
	for ix in range(0,N[0]):
		for iy in range(0,N[1]):
			tau[ix,iy,:,:] = stress[ix,iy,:,:] - compute_behaviour(mu0, Lambda0, strain[ix,iy,:,:])

	#calcul de epsilon a l'etape suivante
	tf_tau = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d))
	tf_gamma0_tau = np.zeros(np.concatenate((N,d,d),axis=None),dtype=np.complex)
	#produit doublement contracté tf_gamma0 : tf_tau
	for ix in range(0,N[0]):
		for iy in range(0,N[1]):
			for iid in range(0,d):
				for jd in range(0,d):
					for kd in range(0,d):
						for hd in range(0,d):
							tf_gamma0_tau[ix,iy,iid,jd] = tf_gamma0_tau[ix,iy,iid,jd] + tf_gamma0[ix,iy,iid,jd,kd,hd] * tf_tau[ix,iy,kd,hd]
	strain_new = strain_macro - np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(tf_gamma0_tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d)))

	#compute residual
	residual = np.linalg.norm(strain_new-strain)
	strain = strain_new

	# update strain
	for ix in range(0,N[0]):
		for iy in range(0,N[1]):
			stress[ix,iy,:,:] = compute_behaviour(mu[ix,iy], Lambda[ix,iy], strain[ix,iy,:,:])
	print("Iteration %ld" % iteration)
	print("Residual %.8e" % residual)
	iteration = iteration + 1       #compter le nombre d'iteration

plt.imshow(strain[:,:,0,0], interpolation='none', origin='lower')
plt.colorbar()
plt.show()
#print(stress)
#print(strain_new)
