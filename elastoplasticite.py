# -*-coding:utf-8 -*
import numpy as np
import scipy as sp
import math as m
import fct
import matplotlib.pyplot as plt
import sys
import time


d = fct.d
Lx = fct.Lx
N = fct.N
Ntotal = fct.Ntotal
mu = fct.mu
Lambda = fct.Lambda
sigma_y = fct.sigma_y
mu0 = fct.mu0
Lambda0 = fct.Lambda0
K = fct.K
mu_0 = fct.mu_0
Lambda_0 = fct.Lambda_0
mu_1 = fct.mu_1
Lambda_1 = fct.Lambda_1
tf_gamma0 = fct.tf_gamma0


# time increments
T = 0.001
NT = 11
dt = T/(NT-1)
t = np.linspace(0,T,NT)
if(len(sys.argv) > 1):
	strain_macro = np.load(sys.argv[1])
	print(strain_macro)
else:
	strain_macro = np.zeros((NT,d,d))
	strain_macro[:,0,1] = np.linspace(0,0.5*T,NT)
	strain_macro[:,1,0] = strain_macro[:,0,1]
	#strain_macro[:,1,0] = strain_macro[:,0,1]
	# pour calculer A
	# ne pas oublier de mettre NT = 2
	# strain_macro[1,:,:] = [[1e-6,0],[0,0]]
	# strain_macro[1,:,:] = [[0,1e-6],[1e-6,0]]
	# strain_macro[1,:,:] = [[0,0],[0,1e-6]]


# variables
strain = np.zeros(np.concatenate((NT,N,d,d),axis=None))
stress = np.zeros(np.concatenate((NT,N,d,d),axis=None))
strain_pl_equiv = np.zeros(np.concatenate((NT,N),axis=None))
strain_pl = np.zeros(np.concatenate((NT,N,d,d),axis=None))


# boundary conditions
#strain_macro = [np.array([[0.01,0],[0,0]]), np.array([[0,0],[0,0.01]]), np.array([[0,0.005],[0.005,0]]), np.array([[0,0.005],[0.005,0]])]

# algorithm
for tn in range(1,NT):

	# strain_nminus = strain.copy()
	# stress_nminus = stress.copy()
	# strain_pl_equiv_nminus = strain_pl_equiv.copy()
	# strain_pl_nminus = strain_pl.copy()

	residual = 1
	iteration = 0
	while residual > 1e-6:

		#compute polarization stress
		tau = np.zeros(np.concatenate((N,d,d),axis=None))
		for ix in range(0,N[0]):
			for iy in range(0,N[1]):
				tau[ix,iy,:,:] = stress[tn,ix,iy,:,:] - fct.compute_behaviour_elastic(mu0, Lambda0, strain[tn,ix,iy,:,:])

		tf_tau = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d))
		tf_gamma0_tau = np.zeros(np.concatenate((N,d,d),axis=None),dtype=np.complex)

		#produit doublement contracté tf_gamma0 : tf_tau
		for ix in range(0,N[0]):
			for iy in range(N[1]):
				tf_gamma0_tau[ix,iy,:,:] = np.einsum('ijkl,kl->ij', tf_gamma0[ix,iy,:,:,:,:],tf_tau[ix,iy,:,:])


		strain_new = strain_macro[tn,:,:] - np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(tf_gamma0_tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d)))
		#strain_new = strain_macro_tn - np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(tf_gamma0_tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d)))

		# Compute residual
		residual = np.linalg.norm(strain_new-strain[tn,:,:,:,:])
		strain[tn,:,:,:,:] = strain_new

		# Update stress
		for ix in range(0,N[0]):
			for iy in range(0,N[1]):
				(stress[tn,ix,iy,:,:],strain_pl_equiv[tn,ix,iy],strain_pl[tn,ix,iy,:,:]) = fct.compute_behaviour_elastoplastic(dt, mu[ix,iy], Lambda[ix,iy], sigma_y[ix,iy], K[ix,iy], strain[tn-1,ix,iy,:,:], strain[tn,ix,iy,:,:], stress[tn-1,ix,iy,:,:], strain_pl_equiv[tn-1,ix,iy],strain_pl[tn-1,ix,iy,:,:])
				if(tn == 1 and strain_pl_equiv[tn,ix,iy] > 0):
					print("ERROR: DECREASE LOAD INCREMENTS\n")
					exit(-1)
		# print("Iteration %ld" % iteration)
		# print("Residual %.8e" % residual)
		iteration = iteration + 1       #compter le nombre d'iteration
	#print("TIME ITER %ld FINISHED" % tn)

np.save("strain", strain)
np.save("stress", stress)
np.save("strain_pl_equiv", strain_pl_equiv)
np.save("strain_pl", strain_pl)


#for ix in range(0,d):
#	for iy in range(0,d):
#		plt.imshow(stress[4,:,:,ix,iy].T, interpolation='none', origin='lower')
#		plt.colorbar()
#		plt.show()

#print(strain_macro[NT-1,:,:])
#for tn in range(0,NT):
#	for ix in range(0,d):
#		for iy in range(0,d):
#			plt.imshow(stress[tn,:,:,ix,iy], interpolation='none', origin='lower')
#			plt.colorbar()
#			plt.show()
#m = np.zeros(N)
#for ix in range(0,N[0]):
#	for iy in range(0,N[1]):
#		#print(np.mean(stress[0,0,...]))
#		m[ix,iy] = np.mean(strain[NT-1,ix,iy,...])
#		mix = m.reshape(N[0]*N[1])
#plt.plot(mix)
#plt.show()

#plt.imshow(strain_pl_equiv[NT-1,:,:].T, interpolation='none', origin='lower')
#plt.colorbar()
#plt.show()

strain_pl_equiv_avg = strain_pl_equiv.copy()

for jd in range(1,d+1):
	strain_pl_equiv_avg = strain_pl_equiv_avg.sum(axis=1)
	strain_pl_equiv_avg = strain_pl_equiv_avg / Ntotal
	plt.plot(np.linspace(0, T, NT), strain_pl_equiv_avg)
	plt.show()



#plt.colorbar()
#plt.show()
#print(strain_pl_equiv.shape)
#aff = strain_



#print(iteration)
#print(time.perf_counter())
