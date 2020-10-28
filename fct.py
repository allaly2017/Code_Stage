# -*-coding:utf-8 -*
import numpy as np
import scipy as sp
import math as m
import matplotlib.pyplot as plt
import sys


# Trace(A) = np.einsum('ii',A)

# Produit doublement contracté tenseur ordre 2 par tenseur ordre 2 -> scalaire
# np.einsum('ij,ji',A,A)

# Produit doublement contracté tenseur ordre 4 par tenseur ordre 2 -> tenseur ordre 2
# np.einsum('ijkl,kl->ij', B,A)

# Produit doublement contracté tenseur ordre 4 par tenseur ordre 4 -> tenseur ordre 4
# np.einsum('ijkl,klop->ijop',AC)

d = 2

# domain definition
Lx = 1                            # domain size in x direction
N = np.zeros(d,dtype=np.int)      # number of voxels in x direction
N[0] = 15
N[1] = 15
Ntotal = N.prod()


def convert_elastic_modulii(Young, Poisson):
	mu = Young/(2*(1+Poisson))
	Lambda = Young*Poisson/((1+Poisson)*(1-2*Poisson))
	return (mu, Lambda)

def delta(a,b):
  	if (a==b):
  		kronecker = 1.
  	else:
  		kronecker = 0.
  	return kronecker


# material properties
(mu_0, Lambda_0) = convert_elastic_modulii(400000, 0.33)
(mu_1, Lambda_1) = convert_elastic_modulii(210000, 0.33)

mu = np.zeros(N)
Lambda = np.zeros(N)
sigma_y = np.zeros(N)
K = np.zeros(N)
for ix in range(0,N[0]):
	for iy in range(0,N[1]):
		if(ix>=(N[0]/3) and iy>=(N[1]/3) and ix<(N[0]*2/3) and iy<(N[1]*2/3)):
			mu[ix,iy] = mu_0
			Lambda[ix,iy] = Lambda_0
			sigma_y[ix,iy] = 1e10 # inclusion remains elastic
			K[ix,iy] = 1
		else:
			mu[ix,iy] = mu_1
			Lambda[ix,iy] = Lambda_1
			sigma_y[ix,iy] = 100
			K[ix,iy] = 1000

# wisely choose reference material parameters
mu0 = 0.5*(mu.max() + mu.min())
Lambda0 = 0.5*(Lambda.max() + Lambda.min())

#Green's operator
tf_gamma0 = np.zeros(np.concatenate((N,d,d,d,d),axis=None))
for ix in range(0,N[0]):
	for iy in range(0,N[1]):
		ksi = np.zeros(d)
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
							tf_gamma0[ix,iy,iid,jd,kd,hd] = (delta(kd,iid)*ksi[hd]*ksi[jd] + delta(hd,iid)*ksi[kd]*ksi[jd] + delta(kd,jd)*ksi[hd]*ksi[iid] + delta(hd,jd)*ksi[kd]*ksi[iid])/(4*mu0*ksi_2) - ((Lambda0+mu0)/(mu0*(Lambda0+2*mu0)))*(ksi[iid]*ksi[jd]*ksi[kd]*ksi[hd])/ksi_4


# material subroutine for small strain elastic behaviour
def compute_behaviour_elastic(mu, Lambda, strain):
	trace_strain = np.einsum('ii',strain)
	return 2*mu*strain +  Lambda*trace_strain*np.eye(strain.shape[0])

def ecrou_iso(sigma_y, K, variable):
    return sigma_y + K*variable


# material subroutine for small strain elasto-plastic behaviour
def compute_behaviour_elastoplastic(dt, mu, Lambda, sigma_y, K, \
	strain_nminus, strain, stress_nminus, strain_pl_equiv_nminus, strain_pl_nminus):

	d = strain.shape[0]

	# contrainte test
	stress_trial = stress_nminus + compute_behaviour_elastic(mu, Lambda, strain - strain_nminus)

	# calcul du Tenseur deviatorique s test
	s_trial = stress_trial - (1./3)*np.einsum('ii',stress_trial)*np.eye(d)

	# Equation de la surface de charge
	#f_trial = m.sqrt((3./2)*produit_contracte(s_trial, s_trial))
	f_trial = m.sqrt((3./2)*np.einsum('ij,ji',s_trial,s_trial))

	# Seuil de plasticite
	strain_pl_equiv_trial = strain_pl_equiv_nminus
	seuil_plasticite = ecrou_iso(sigma_y, K, strain_pl_equiv_trial)

	# Test sur la surface de charge
	if(f_trial <= seuil_plasticite):
		#stress = stress_trial
		#tau_strain_pl_equiv_nplus = strain_pl_equiv_trial
		return (stress_trial,strain_pl_equiv_trial,strain_pl_nminus)
	else:
		r_trial = (3./2)*(s_trial/f_trial)
		rhat_trial = compute_behaviour_elastic(mu, Lambda, r_trial)
		#s_r = produit_contracte(s_trial,rhat_trial)
		#r_r = produit_contracte(rhat_trial,rhat_trial)
		s_r = np.einsum('ij,ji',s_trial,rhat_trial)
		r_r = np.einsum('ij,ji',rhat_trial,rhat_trial)

		res = sys.float_info.max
		rate_strain_pl_equiv = 0.
		it = 0
		while (res > 1e-8):
			strain_pl_equiv = strain_pl_equiv_nminus + dt*rate_strain_pl_equiv
			seuil_plasticite = ecrou_iso(sigma_y, K, strain_pl_equiv)
			G = (f_trial)**2 - 3*dt*rate_strain_pl_equiv*s_r + (3./2)*(dt**2)*(rate_strain_pl_equiv**2)*r_r - seuil_plasticite**2
			G_prime = -3*dt*s_r + 3*(dt**2)*rate_strain_pl_equiv*r_r - 2*dt*K*seuil_plasticite
			delta_rate_strain_pl_equiv = - G/G_prime
			rate_strain_pl_equiv_new = rate_strain_pl_equiv + delta_rate_strain_pl_equiv

			sub_it = 0
			while(True):
				rate_strain_pl_equiv_new = rate_strain_pl_equiv + delta_rate_strain_pl_equiv
				strain_pl_equiv = strain_pl_equiv_nminus + dt*rate_strain_pl_equiv_new
				seuil_plasticite = ecrou_iso(sigma_y, K, strain_pl_equiv)
				G = (f_trial)**2 - 3*dt*rate_strain_pl_equiv_new*s_r + (3./2)*(dt**2)*(rate_strain_pl_equiv_new**2)*r_r - seuil_plasticite**2
				res_new = abs(G)

				sub_it = sub_it + 1
				# print("res_new : %e" % res_new)
				# print("rate_strain_pl_equiv_new : %e" % rate_strain_pl_equiv_new)
				if(res_new < res and rate_strain_pl_equiv_new >= 0):
					# print("sub converged")
					break
				elif(sub_it > 1000):
					print("sub diverged")
					exit(0)
				else:
					delta_rate_strain_pl_equiv = delta_rate_strain_pl_equiv*2./3
			res = res_new
			rate_strain_pl_equiv = rate_strain_pl_equiv_new

			it = it + 1
			if(it > 1000):
				print("diverged")
				exit(0)

		strain_pl = strain_pl_nminus + dt*rate_strain_pl_equiv*r_trial
		return(stress_trial - dt*rate_strain_pl_equiv*rhat_trial, strain_pl_equiv, strain_pl)




def elasticity(mu,Lambda,mu0,Lambda0,d,N,strain_macro):

	strain = np.zeros(np.concatenate((N,d,d),axis=None))

	#contrainte à la première iteration
	stress = np.zeros(np.concatenate((N,d,d),axis=None))
	for ix in range(0,N[0]):
		for iy in range(0,N[1]):
			stress[ix,iy,:,:] = compute_behaviour_elastic(mu[ix,iy], Lambda[ix,iy], strain[ix,iy,:,:])

	# algorithm
	residual = 1
	iteration = 0
	while residual > 1e-6:

		#compute polarization stress
		tau = np.zeros(np.concatenate((N,d,d),axis=None))
		for ix in range(0,N[0]):
			for iy in range(0,N[1]):
				tau[ix,iy,:,:] = stress[ix,iy,:,:] - compute_behaviour_elastic(mu0, Lambda0, strain[ix,iy,:,:])

		#calcul de epsilon a l'etape suivante
		tf_tau = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d))
		tf_gamma0_tau = np.zeros(np.concatenate((N,d,d),axis=None),dtype=np.complex)

		#produit doublement contracté tf_gamma0 : tf_tau
		for ix in range(0,N[0]):
			for iy in range(N[1]):
				tf_gamma0_tau[ix,iy,:,:] = np.einsum('ijkl,kl->ij', tf_gamma0[ix,iy,:,:,:,:],tf_tau[ix,iy,:,:])

		strain_new = strain_macro - np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(tf_gamma0_tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d)))

		#compute residual
		residual = np.linalg.norm(strain_new-strain)
		strain = strain_new

		# update strain
		for ix in range(0,N[0]):
			for iy in range(0,N[1]):
				stress[ix,iy,:,:] = compute_behaviour_elastic(mu[ix,iy], Lambda[ix,iy], strain[ix,iy,:,:])
		#print("Iteration %ld" % iteration)
		#print("Residual %.8e" % residual)
		iteration = iteration + 1       #compter le nombre d'iteration

		#plt.imshow(strain[:,:,0,0], interpolation='none', origin='lower')
		#plt.colorbar()
		#plt.show()
	return strain
