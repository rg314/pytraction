# % Copyright (C) 2010 - 2019, Sabass Lab
# %
# % This program is free software: you can redistribute it and/or modify it 
# % under the terms of the GNU General Public License as published by the Free
# % Software Foundation, either version 3 of the License, or (at your option) 
# % any later version. This program is distributed in the hope that it will be 
# % useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General 
# % Public License for more details. You should have received a copy of the 
# % GNU General Public License along with this program.
# % If not, see <http://www.gnu.org/licenses/>.


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %DESCRIPTION
# %Function for calculating regularization parameter using Bayesian method

# %------------------
# %FUNCTION ARGUMENTS 
# %beta: 1/variance of noise 
# %fuu: displacement vector in Fourior space
# %Ftux: x component of displacement matrix in Fourior space
# %Ftuy: y component of displacement matrix in Fourior space
# %E: Young's modulus
# %s: Poisson's ratio
# %cluster_size: grid spacing in pixels
# %grid_mat: regular grid with size i_max*j_max 
# %u: displacement vectors on grid
# %i_max, j_max: sizes of grid
# %X: matrix between displacement and force in Fourior space
# %sequence: set to 1 if only maximum evidence parameter should be returned
# %------------------

# %------------------
# %FUNCTION OUTPUTS
# %lambda_2: optimal regularization parameter
# %evidencep: matrix for regularization parameter and its value of
# %          logevidence around the optimal regularization parameter 
# %evidence_one: value of logevidence at the optimal regularization parameter
# %------------------


# function [lambda_2 evidencep evidence_one]  = optimal_lambda(beta,fuu,Ftux,Ftuy,E,s,cluster_size,i_max, j_max,X,sequence)

import numpy as np 
from functools import partial

from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import cholesky
# from scikits.sparse.cholmod import cholesky
import scipy.optimize as optimize
import time 

from pytraction.reg_fourier import reg_fourier_tfm

def minus_logevidence(alpha, beta, C_a, BX_a, X, fuu, constant, Ftux,Ftuy,E,s,cluster_size,i_max, j_max):
    aa = X.shape
    LL = alpha/beta
    _,_,_,_,Ftfx, Ftfy = reg_fourier_tfm(Ftux,Ftuy,LL,E,s,cluster_size,i_max, j_max, slim=True)
    fxx = Ftfx.reshape(i_max*j_max,1)
    fyy = Ftfy.reshape(i_max*j_max,1)

    f = np.array([fxx,fyy]).T.flatten()
    f = np.expand_dims(f, axis=1)

    A = alpha*csr_matrix(C_a) + BX_a
    L = cholesky(csr_matrix(A).toarray())
    logdetA = 2*np.sum(np.log(np.diag(L)))

    Xf_u = X*f-fuu
    idx = Xf_u.shape[0]//2
    Ftux1= Xf_u[:idx]
    Ftuy1= Xf_u[idx:]

    ff = np.sum(np.sum(Ftfx*np.conj(Ftfx) + Ftfy*np.conj(Ftfy)))/(0.5*aa[1])
    # ff = np.real(ff)
    uu = np.sum(np.sum(Ftux1*np.conj(Ftux1) + Ftuy1*np.conj(Ftuy1)))/(0.5*aa[0])
    # uu = np.real(uu)

    evidence_value = -0.5*(-alpha*ff-beta*uu -logdetA +aa[1]*np.log(alpha)+constant)
    return evidence_value


def optimal_lambda(beta,fuu,Ftux,Ftuy,E,s,cluster_size,i_max, j_max,X,sequence):
    aa = X.shape
    c = np.ones((aa[1]))
    C = spdiags(c, (0), aa[1], aa[1])
    XX = csr_matrix(X).T*csr_matrix(X)
    BX_a = beta*csr_matrix(XX)/aa[1]*2
    C_a = C/aa[1]*2
    constant = aa[0]*np.log(beta)-aa[0]*np.log(2*np.pi)

    # Golden section search method to find alpha at minimum of -log(Evidence)
    # setting the range of parameter search. Change if maximum can not be found in your data
    alpha1 =1e-6
    alpha2 =1e6

    print('Optimizing Lambda')
    target = partial(minus_logevidence, beta=beta, C_a=C_a, BX_a=BX_a, X=X, fuu=fuu, constant=constant, Ftux=Ftux,Ftuy=Ftuy,E=E,s=s,cluster_size=cluster_size,i_max=i_max, j_max=j_max)
    start = time.time()
    alpha_opt = optimize.fminbound(target, alpha1, alpha2, disp=3)
    end = time.time()
    print(f'Time taken {end-start} s')

    evidence_one = -target(alpha_opt)
    lambda_2 = alpha_opt/beta

    return lambda_2,None,evidence_one





 
 



