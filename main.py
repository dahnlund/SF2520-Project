# %% Part 2
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu
from matplotlib import pyplot as plt


def create_DAE_system(A, M, N, dtau, epsilon: float = 0):
    I_A1 = np.eye(M - 1)
    I_tot = np.zeros((M + N - 1, M + N - 1))
    I_tot[: M - 1, : M - 1] = I_A1
    u0 = np.ones((M + N - 1, 1))

    if epsilon != 0:
        I_tot[M - 1 :, M - 1 :] = 1 / epsilon * np.eye(N)
        I_tot = csc_matrix(I_tot)
        LHS = splu(eye(M + N - 1, format="csc") - dtau * I_tot.dot(A))
        RHS = lambda uk: uk

    else:
        I_tot = csc_matrix(I_tot)
        u0[: M - 1] = 1
        LHS = splu(I_tot - dtau * A)
        RHS = lambda uk: I_tot.dot(uk)

    return LHS, RHS, u0, dtau


def impl_euler(LHS, RHS, u0, dtau):
    """Implicit Euler"""
    tau = np.arange(dtau, 1, dtau)
    saved_u = np.zeros((len(u0), len(tau)+1))
    saved_u[:, 0] = u0[:, 0]

    uk = u0
    for i, _ in enumerate(tau):
        u_new = LHS.solve(RHS(uk))
        saved_u[:, i+1] = u_new[:, 0]
        uk = u_new

    saved_u = np.vstack([1/3*(4*saved_u[0,:]-saved_u[1,:]),saved_u, 1/3*(4*saved_u[-1,:]-saved_u[-2,:])])
    return saved_u


def v(z):
    return 1 - 4 * (z - 1 / 2) ** 2


def create_A(M, N, z, eta, gamma, alpha):
    dz = 1 / M
    A1_data = np.array(
        [
            eta * np.ones(M - 1) / (v(z[1:M] + dz) * dz**2),
            -2 * eta * np.ones(M - 1) / (v(z[1:M]) * dz**2),
            eta * np.ones(M - 1) / (v(z[1:M] - dz) * dz**2),
        ]
    )
    A1 = sp.spdiags(A1_data, np.array([-1, 0, 1]), format="csc")

    # Adjust for first boundary condition:
    A1[0, 0] = eta / (v(z[1]) * dz**2) * (-2 / 3)
    A1[0, 1] = eta / (v(z[1]) * dz**2) * (2 / 3)

    A2_data = np.array(
        [
            np.ones(N - 1) / (dz**2),
            -2 * np.ones(N - 1) / (dz**2) - gamma,
            np.ones(N - 1) / (dz**2),
        ]
    )
    A2 = sp.spdiags(A2_data, np.array([-1, 0, 1]), format="csc")
    A2[-1, -1] = 1 / (dz**2) * (-2 / 3) - gamma
    A2[-1, -2] = 1 / (dz**2) * (2 / 3)

    b1 = np.zeros((M - 1, 1))
    b1[-1] = -1 / dz
    b2 = np.zeros((N - 1, 1))
    b2[0] = -alpha / dz
    a = (1 + alpha) / dz
    e1 = np.zeros((M - 1, 1))
    e1[-1] = eta / ((dz**2) * v((M - 1) * dz))

    e2 = np.zeros((N - 1, 1))
    e2[0] = 1 / (dz**2)

    block1 = sp.hstack([A1, csc_matrix(e1), csc_matrix(np.zeros((M - 1, N - 1)))])
    block2 = sp.hstack([csc_matrix(b1.T), csc_matrix(a), csc_matrix(b2.T)])
    block3 = sp.hstack([csc_matrix(np.zeros((N - 1, M - 1))), csc_matrix(e2), A2])
    A = sp.vstack([block1, block2, block3])
    return A


def main(eta=0.2, gamma=100, alpha=0.2, w=0.3, M=1000, epsilon=0, dtau=0.01):
    N = round(M * w)
    z = np.linspace(0, 1 + w, M + N + 1)
    A = create_A(M, N, z, eta, gamma, alpha)

    dtau = 0.01
    u = impl_euler(*create_DAE_system(A, M, N, dtau, epsilon=epsilon))
    tau = np.arange(dtau, 1, dtau)

    n_traces = 10
    colors = plt.get_cmap("viridis")(np.linspace(0.8, 0, n_traces))

    title = f"{eta=} {gamma=} {alpha=} {w=} {M=} {epsilon=} {dtau=}"

    for i in range(n_traces):
        j = int(i * (len(tau) / n_traces))
        plt.plot(z, u[:, j], label=f"tau={tau[j]:.2f}", color=colors[i])
    # plt.legend()
    plt.ylim(0, 1.05)
    plt.grid()
    plt.title(title, fontsize=9.5)
    plt.savefig(f"figures/{title}.png")
    plt.clf()


if __name__ == "__main__":
    for eta in [0.2, 0.4, 0.6]:
        main(eta=eta)

# %%
