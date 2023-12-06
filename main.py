# %% Part 2
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import get_ipython

if get_ipython() is not None:  # Only run while in Jupyter
    get_ipython().run_line_magic("matplotlib", "qt")


# %%
class DAEModule:
    def __init__(self, A, A1, A2, dtau):
        self.A = A
        self.A1 = A1
        self.A2 = A2
        self.dtau = dtau

    def solve(self, epsilon: float = 0):
        M = self.A1.shape[0] + 1
        I_A1 = np.eye(self.A1.shape[0])
        I_tot = np.zeros(self.A.shape)
        I_tot[: M - 1, : M - 1] = I_A1

        if epsilon != 0:
            I_tot[M - 1 :, M - 1 :] = 1 / epsilon * np.eye(self.A2.shape[0] + 1)
            I_tot = csc_matrix(I_tot)
            u0 = np.ones((self.A.shape[1], 1))
            LHS = splu(
                eye(self.A.shape[0], format="csc") - self.dtau * I_tot.dot(self.A)
            )
            RHS = lambda uk: uk

        else:
            I_tot = csc_matrix(I_tot)
            u0 = np.zeros((self.A.shape[1], 1))
            u0[: M - 1] = 1
            LHS = splu(I_tot - self.dtau * self.A)
            RHS = lambda uk: I_tot.dot(uk)

        return self.impl_euler(LHS, RHS, u0)

    def impl_euler(self, LHS, RHS, u0):
        """Implicit Euler"""
        tau = np.arange(self.dtau, 1, self.dtau)
        saved_u = np.zeros((len(u0), len(tau)))
        saved_u[:, 0] = u0[:, 0]

        uk = u0
        for i, _ in enumerate(tau):
            u_new = LHS.solve(RHS(uk))
            saved_u[:, i] = u_new[:, 0]
            uk = u_new
        return tau, saved_u


def v(z):
    return 1 - 4 * (z - 1 / 2) ** 2


# %%
eta = 0.2
gamma = 100
alpha = 0.2
w = 0.3
M = 1000
dz = 1 / M
N = int(round(w / dz))
z = np.linspace(0, 1 + w, M + N + 1)
A1_data = np.array(
    [
        eta * np.ones(z[1:M].shape) / (v(z[1:M] + dz) * dz**2),
        -2 * eta * np.ones(z[1:M].shape) / (v(z[1:M]) * dz**2),
        eta * np.ones(z[1:M].shape) / (v(z[1:M] - dz) * dz**2),
    ]
)
A1 = sp.spdiags(A1_data, np.array([-1, 0, 1]), format="csc")

# Adjust for first boundary condition:
A1[0, 0] = eta / (v(z[1]) * dz**2) * (-2 / 3)
A1[0, 1] = eta / (v(z[1]) * dz**2) * (2 / 3)

A2_data = np.array(
    [
        np.ones(z[M + 1 : M + N].shape) / (dz**2),
        -2 * np.ones(z[M + 1 : M + N].shape) / (dz**2) - gamma,
        np.ones(z[M + 1 : M + N].shape) / (dz**2),
    ]
)
A2 = sp.spdiags(A2_data, np.array([-1, 0, 1]), format="csc")
A2[-1, -1] = 1 / (dz**2) * (-2 / 3) - gamma
A2[-1, -2] = 1 / (dz**2) * (2 / 3)

b1 = np.zeros((A1.shape[0], 1))
b1[-1] = -1 / dz
b2 = np.zeros((A2.shape[0], 1))
b2[0] = -alpha / dz
a = (1 + alpha) / dz
e1 = np.zeros((A1.shape[0], 1))
e1[-1] = eta / ((dz**2) * v((M - 1) * dz))

e2 = np.zeros((A2.shape[0], 1))
e2[0] = 1 / (dz**2)

# %% Concatenate everything to block matrix system
block1 = sp.hstack(
    [A1, csc_matrix(e1), csc_matrix(np.zeros((A1.shape[0], A2.shape[1])))]
)
block2 = sp.hstack([csc_matrix(b1.T), csc_matrix(a), csc_matrix(b2.T)])
block3 = sp.hstack(
    [csc_matrix(np.zeros((A2.shape[0], A1.shape[1]))), csc_matrix(e2), A2]
)
A = sp.vstack([block1, block2, block3])

# %%
dtau = 0.001
model = DAEModule(A, A1, A2, dtau)

tau, u = model.solve()
_, u_reg = model.solve(0.01)

# %% 3D plot

Z, TAU = np.meshgrid(z[1:-1], tau)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Z, TAU, u_reg.T)
ax.plot_surface(Z, TAU, u.T)
ax.legend(("u", "u with reg"))
plt.show()


# %%
fig, ax = plt.subplots()

# Plot the initial lines
(line1,) = ax.plot(z[1:-1], u[:, 0], label="u")
(line2,) = ax.plot(z[1:-1], u_reg[:, 0], label="u with reg")


def update(frame):
    line1.set_ydata(u_reg[:, frame])
    line2.set_ydata(u[:, frame])

    ax.legend()
    return (line1, line2)


ani = FuncAnimation(fig, update, frames=range(u_reg.shape[1]), interval=1)
plt.show()

# %%
