import numpy as np 
from functools import partial

from scipy.sparse import spdiags, csr_matrix
import scipy.optimize as optimize

import time 

from pytraction.utils import sparse_cholesky
from pytraction.fourier import reg_fourier_tfm

def minus_logevidence(alpha, beta, C_a, BX_a, X, fuu, constant, Ftux,Ftuy,E,s,cluster_size,i_max, j_max):
    aa = X.shape
    LL = alpha/beta
    _,_,_,_,Ftfx, Ftfy = reg_fourier_tfm(Ftux,Ftuy,LL,E,s,cluster_size,i_max, j_max, slim=True)
    fxx = Ftfx.reshape(i_max*j_max,1)
    fyy = Ftfy.reshape(i_max*j_max,1)

    f = np.array([fxx,fyy]).T.flatten()
    f = np.expand_dims(f, axis=1)

    A = alpha*csr_matrix(C_a) + BX_a
    
    L = sparse_cholesky(csr_matrix(A)).toarray()
    logdetA = 2*np.sum(np.log(np.diag(L)))

    Xf_u = X*f-fuu
    idx = Xf_u.shape[0]//2
    Ftux1= Xf_u[:idx]
    Ftuy1= Xf_u[idx:]

    ff = np.sum(np.sum(Ftfx*np.conj(Ftfx) + Ftfy*np.conj(Ftfy)))/(0.5*aa[1])
    # ff = np.real(ff)
    uu = np.sum(np.sum(Ftux1*np.conj(Ftux1) + Ftuy1*np.conj(Ftuy1)))/(0.5*aa[0])
    # uu = np.real(uu)

    evidence_value = np.real(-0.5*(-alpha*ff-beta*uu -logdetA +aa[1]*np.log(alpha)+constant))
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
    alpha1 = 1e-6
    alpha2 = 1e6

    target = partial(minus_logevidence, beta=beta, C_a=C_a, BX_a=BX_a, X=X, fuu=fuu, 
                    constant=constant, Ftux=Ftux, Ftuy=Ftuy, E=E, s=s, 
                    cluster_size=cluster_size, i_max=i_max, j_max=j_max)
    alpha_opt = optimize.fminbound(target, alpha1, alpha2, disp=3)

    evidence_one = -target(alpha_opt)
    lambda_2 = alpha_opt / beta

    return lambda_2, None, evidence_one