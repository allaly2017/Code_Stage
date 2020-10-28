# -*-coding:utf-8 -*
import numpy as np
import scipy.optimize as sp
import math as m
import matplotlib.pyplot as plt
import sys
#import elastoplasticite
import fct
#import generator
import os
import random
import subprocess
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import time

#import pod_compressor
#import glob
#import re
#from sklearn.decomposition import PCA
#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split


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

#print(strain_macro.shape)

W = np.load("strain_pl_vectors_6.npy")
nb_modes = W.shape[0]
W = W.reshape(np.concatenate((nb_modes,N,d,d),axis=None))

# boundary conditions
strain_macro_A = 0.0001*np.array([[[[1,0],[0,0]],[[0,0.5],[0.5,0]]],
                    [[[0,0.5],[0.5,0]], [[0,0],[0,1]]]])

#Tenseur de Localisation
A = np.zeros(np.concatenate((N,d,d,d,d),axis=None))
for iid in range(0,d):
    for jd in range(0,d):
        strain = fct.elasticity(mu,Lambda,mu0,Lambda0,d,N,strain_macro_A[iid,jd])
        A[:,:,:,:,iid,jd] = strain/0.0001

# Constante d'elasticite
C = np.zeros(np.concatenate((N,d,d,d,d), axis=None))
for ix in range(0,N[0]):
    for iy in range(0,N[1]):
        for iid in range(0,d):
            for jd in range(0,d):
                for kd in range(0,d):
                    for hd in range(0,d):
                        C[ix,iy,iid,jd,kd,hd] = Lambda[ix,iy]*fct.delta(iid,jd)*fct.delta(kd,hd) + 2*mu[ix,iy]*fct.delta(iid,kd)*fct.delta(jd,hd)

# Calcul de l'opérateur D
D = np.zeros(np.concatenate((nb_modes,N,d,d), axis=None))
for k in range(0,nb_modes):
	tau = np.einsum('...ijkl,...kl->...ij', C, W[k,...])
	tf_tau = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d))
	tf_gamma0_tau = np.einsum('...ijkl,...kl->...ij', tf_gamma0, tf_tau)
	D[k] = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(tf_gamma0_tau, axes=range(0,d)), s=N, axes=range(0,d)), axes=range(0,d)))

C_D_W = np.einsum('...ijkl,...kl->...ij', C, D-W)

# Calcul de l'opérateur DD
DD = np.zeros((nb_modes,nb_modes))
for k in range(0,nb_modes):
    for l in range(0,nb_modes):
        pd = np.einsum('...ij,...ij->...', C_D_W[k,...], W[l,...])
        DD[l,k] = pd.mean()



def compute_rate_alpha_lambda_pl(alpha_minus, alpha, dt, W, lambda_pl_minus):
	rate_alpha = (alpha - alpha_minus)/dt
	rate_strain_pl = np.einsum('k,k...->...', rate_alpha, W)
	rate_strain_pl_equiv = np.sqrt((2./3)*np.einsum('...ij,...ij->...', rate_strain_pl, rate_strain_pl))
	lambda_pl = lambda_pl_minus + dt*rate_strain_pl_equiv

	return (rate_alpha, lambda_pl)

def compute_stress_f_macro(d, sigma_y, K, C_A_stm, C_D_W, alpha, lambda_pl):
	stress = C_A_stm + np.einsum('k...,k->...', C_D_W, alpha)
	deviator = stress - (1./3)*np.einsum('...kk,ij->...ij', stress, np.eye(d))
	f_barre = np.sqrt((3./2)*np.einsum('...ij,...ij->...', deviator, deviator))
	f = f_barre - fct.ecrou_iso(sigma_y, K, lambda_pl)
	f_macro = f.max()

	return (stress, f_macro)

#fonction à minimiser
def g(x):
	alpha = x[0:nb_modes]
	lambda_pl_macro = x[nb_modes]

	rate_alpha, lambda_pl = compute_rate_alpha_lambda_pl(alpha_minus, alpha, dt, W, lambda_pl_minus)
	stress, f_macro = compute_stress_f_macro(d, sigma_y, K, C_A_stm, C_D_W, alpha, lambda_pl)

	#print(alpha)
	print("%e -> %e" % (lambda_pl_macro, f_macro))

	#print(S)
	#print(DD)

	S_rate_alpha = np.einsum('k,k->', S, rate_alpha)
	DD_alpha_rate_alpha = np.einsum('lk,k,l->', DD, alpha, rate_alpha)

	print("DISS %e" % (S_rate_alpha + DD_alpha_rate_alpha))
	# exit(0)
	return -(S_rate_alpha + DD_alpha_rate_alpha) + lambda_pl_macro*f_macro

stress = np.zeros(np.concatenate((N,d,d),axis=None))
alpha = np.zeros(nb_modes)
lambda_pl = np.zeros(N)
lambda_pl_macro = 0
strain_pl_equiv = np.zeros(np.concatenate((NT,N),axis=None))

for tn in range(1,NT):

	#strain_macro_tn = strain_macro[0] * tn / NT
	strain_macro_tn = strain_macro[tn,:,:]
	#print(strain_macro_tn)

	alpha_minus = alpha.copy()
	lambda_pl_minus = lambda_pl.copy()

	A_stm = np.einsum('...ijkl,kl->...ij', A, strain_macro_tn)
	C_A_stm = np.einsum('...ijkl,...kl->...ij', C, A_stm)

	S = np.zeros(nb_modes)
	for k in range(0,nb_modes):
		pd = np.einsum('...ij,...ij->...', C_A_stm, W[k,...])
		S[k] = pd.mean()

	stress_trial, f_trial_macro = compute_stress_f_macro(d, sigma_y, K, C_A_stm, C_D_W, alpha, lambda_pl)

	if (f_trial_macro < 0):
		stress = stress_trial
		print('On est passé là une fois')
		print(strain_macro_tn)
		#for ix in range(0,d):
		#	for iy in range(0,d):
		#		plt.imshow(strain[...,ix,iy], interpolation='none', origin='lower')
		#		plt.colorbar()
		#		plt.show()
	else:
		x = np.concatenate((alpha,lambda_pl_macro), axis=None)
		res = sp.minimize(g, x, method='BFGS', jac=None, options={'disp':True})
		x = res.x

		alpha = x[0:nb_modes]
		lambda_pl_macro = x[nb_modes]

		rate_alpha, lambda_pl = compute_rate_alpha_lambda_pl(alpha_minus, alpha, dt, W, lambda_pl_minus)
		stress, f_macro = compute_stress_f_macro(d, sigma_y, K, C_A_stm, C_D_W, alpha, lambda_pl)


	strain_pl_equiv[tn,...] = lambda_pl

	for ix in range(0,d):
		for iy in range(0,d):
			plt.imshow(stress[...,ix,iy], interpolation='none', origin='lower')
			plt.colorbar()
			plt.show()



#for ix in range(0,d):
#	for iy in range(0,d):
#		plt.imshow(strain[:,:,ix,iy], interpolation='none', origin='lower')
#		plt.colorbar()
#		plt.show()

strain_pl_equiv_avg = strain_pl_equiv.copy()
for jd in range(1,d+1):
	strain_pl_equiv_avg = strain_pl_equiv_avg.sum(axis=1)
strain_pl_equiv_avg = strain_pl_equiv_avg / Ntotal
plt.plot(np.linspace(0, T, NT), strain_pl_equiv_avg)
plt.show()


#plt.imshow(lambda_pl, interpolation='none', origin='lower')
#plt.colorbar()
#plt.show()


#for ix in range(0,NT):
#	lambda_pl = lambda_pl.sum(axis=0)
#	lambda_pl = lambda_pl / Ntotal
#	plt.plot(np.linspace(0, T, NT+4), lambda_pl)
#	plt.show()


#print(lambda_pl.shape)
    #exit(0)






#print(time.perf_counter())





















"""
Dans cette partie je calcule :
-la contrainte effective macro de test f_trial_macro
-la dissipation macroscopique
-je presente la fonction à minimiser
-je fais la minimisation avec scipy
???????????? ce qui m'embête c'est les valeurs de alpha_{k} et {l} sont beaucoup
trop grandes pour que je les utilisent pour actualiser les variables afin de
pouvoir passer à une autre itération ????????????????????
"""
#x0 = np.zeros(3)  #Les deux premiers 0 représentent alpha_{k}^{n+1} et alpha_{l}^{n+1} et le dernier est pour lambda
#alpha = np.ones(nb_modes)   #au départ je prend alpha = 1 (non nul)
######################################
#Calcul de la dissipation macroscopique
#somme = np.zeros(np.concatenate((N,d,d), axis=None))
#for k in range(nb_modes):
#    somme = somme + C_Dk_Wk*alpha[k]
#stress_trial = np.einsum('...ijkl,...kl->...ij', C,np.einsum('...ijkl,kl->...ij',A,strain_macro[0])) + somme
#######################################

######################################
#Calcul de la contrainte effective macro de test
#deviator = np.zeros(np.concatenate((N,d,d), axis=None))
#for ix in range(0,N[0]):
#    for iy in range(0,N[1]):
#        deviator[ix,iy,:,:] = stress_trial[ix,iy,:,:] - (1./3)*np.einsum('ii',stress_trial[ix,iy,:,:])*np.eye(d)
#s_trial_macro = deviator.mean()

#f_trial_macro = m.sqrt((2./3)*s_trial_macro*s_trial_macro)
######################################


###########################
#fonction à minimiser
#def g(x):
#    somme_1 = 0
#    somme_2 = 0
#    for l in range(0,nb_modes):
#        somme_1 = somme_1 + S[l]*x[0]
#        for k in range(0,nb_modes):
#            somme_2 = somme_2 + DD[l,k]*x[0]*x[1]
#    return -somme_1 + somme_2 + x0[2]*f_trial_macro
#####################

######################
#optimisation avec scipy
#res = sp.minimize(g, x0, method='BFGS', jac=None, options={'disp':True})
#print(res)
######################
