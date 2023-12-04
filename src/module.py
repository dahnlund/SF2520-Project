import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu


class DAEModule():
    def __init__(self, A, A1, A2, dtau):
        self.A = A
        self.A1 = A1
        self.A2 = A2
        self.dtau = dtau

    def solve(self, epsilon: float = 0):
        M = self.A1.shape[0]+1
        I_A1 = np.eye(self.A1.shape[0])
        I_tot = np.zeros(self.A.shape)
        I_tot[:M-1, :M-1] = I_A1

        if epsilon != 0:
            I_tot[M-1:, M-1:] = 1/epsilon*np.eye(self.A2.shape[0]+1)
            I_tot = csc_matrix(I_tot)
            u0 = np.ones((self.A.shape[1],1))
            LHS = splu(eye(self.A.shape[0], format = 'csc') - self.dtau * I_tot.dot(self.A))
            RHS = lambda uk: uk
        
        else:
            I_tot = csc_matrix(I_tot)
            u0 = np.zeros((self.A.shape[1],1)); u0[:M-1] = 1
            LHS = splu(I_tot - self.dtau * self.A)
            RHS = lambda uk: I_tot.dot(uk)

        return self.impl_euler(LHS, RHS, u0)

    def impl_euler(self, LHS, RHS, u0):
        """Implicit Euler"""
        tau = np.arange(self.dtau, 1, self.dtau)
        saved_u = np.zeros((len(u0), len(tau)))
        saved_u[:, 0] = u0[:, 0]

        uk = u0
        for i,_ in enumerate(tau):
            u_new = LHS.solve(RHS(uk))
            saved_u[:, i] = u_new[:,0]
            uk = u_new
        return tau, saved_u
