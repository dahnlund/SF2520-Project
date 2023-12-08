# %% Part 2
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, eye, sparray
from scipy.sparse.linalg import splu
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import Callable, Any

Array = NDArray[np.float64]

def create_DAE_system(A: Array, M: int, N: int, dtau: float, epsilon: float = 0):
    I_tot_diag = np.zeros(M + N - 1)
    I_tot_diag[: M - 1] = 1  # A_1 part of the matrix
    u0 = np.ones((M + N - 1, 1))

    if epsilon != 0:
        I_tot_diag[M - 1 :] = 1 / epsilon
        I_tot = sp.spdiags([I_tot_diag], diags=0)
        LHS = splu(eye(M + N - 1, format="csc") - dtau * I_tot.dot(A))
        RHS = lambda uk: uk
    else:
        I_tot = sp.spdiags([I_tot_diag], diags=0)
        u0[M - 1 :] -= 1
        LHS = splu(I_tot - dtau * A)
        RHS = lambda uk: I_tot.dot(uk)

    return LHS, RHS, u0


def impl_euler(LHS: Any, RHS: Callable, u0: Array, dtau: float) -> Array:
    """Implicit Euler"""
    tau = np.arange(dtau, 1, dtau)
    saved_u = np.zeros((len(u0), len(tau) + 1))
    saved_u[:, 0] = u0[:, 0]

    uk = u0
    for i, _ in enumerate(tau):
        u_new = LHS.solve(RHS(uk))
        saved_u[:, i + 1] = u_new[:, 0]
        uk = u_new

    saved_u = np.vstack(
        [
            1 / 3 * (4 * saved_u[0, :] - saved_u[1, :]),
            saved_u,
            1 / 3 * (4 * saved_u[-1, :] - saved_u[-2, :]),
        ]
    )
    return saved_u


def v(z: Array) -> Array:
    return 1 - 4 * (z - 1 / 2) ** 2


def create_A(
    M: int, N: int, z: Array, eta, gamma, alpha, analytic_reduction=False, w=0
) -> Array:
    dz = 1 / M

    A1_data = [
        eta * np.ones(M - 1) / (v(z[1:M] + dz) * dz**2),
        -2 * eta * np.ones(M - 1) / (v(z[1:M]) * dz**2),
        eta * np.ones(M - 1) / (v(z[1:M] - dz) * dz**2),
    ]
    A1 = sp.spdiags(A1_data, [-1, 0, 1], format="csc")

    # Adjust for first boundary condition:
    A1[0, 0] = eta / (v(z[1]) * dz**2) * (-2 / 3)
    A1[0, 1] = eta / (v(z[1]) * dz**2) * (2 / 3)

    if analytic_reduction:
        beta = np.tanh(w * np.sqrt(gamma)) * alpha * np.sqrt(gamma)
        A1[-1, -1] = (
            eta / v(z[M - 1]) / (dz**2) * (-2 - 4 * dz * beta) / (3 + 2 * dz * beta)
        )
        A1[-1, -2] = (
            eta / v(z[M - 1]) / (dz**2) * (2 + 2 * dz * beta) / (3 + 2 * dz * beta)
        )
        return A1

    A2_data = [
        np.ones(N - 1) / (dz**2),
        -2 * np.ones(N - 1) / (dz**2) - gamma,
        np.ones(N - 1) / (dz**2),
    ]
    A2 = sp.spdiags(A2_data, [-1, 0, 1], format="csc")
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


def plot3d(z: Array, tau: Array, u: Array):
    TAU, Z = np.meshgrid(tau, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Z, TAU, u, cmap="viridis")
    ax.set_zlim([0, 1.05])
    ax.set_xlim([0, 1.3])
    ax.set_xlabel("z")
    ax.set_ylabel("tau")
    plt.show()


def main(
    eta=0.2,
    gamma=100,
    alpha=0.2,
    w=0.3,
    M=1000,
    epsilon=0,
    dtau=0.01,
    surface_plot=False,
    analytic_reduction=False,
):
    if analytic_reduction == True:
        N = 0
        # Doesn't make a difference, but want create_DAE to use the right RHS
        epsilon = True
    else:
        N = round(M * w)

    z = np.linspace(0, 1 + N / M, M + N + 1)
    A = create_A(M, N, z, eta, gamma, alpha, analytic_reduction, w)

    dtau = 0.01
    LHS, RHS, u0 = create_DAE_system(A, M, N, dtau, epsilon=epsilon)
    u = impl_euler(LHS=LHS, RHS=RHS, u0=u0, dtau=dtau)
    if analytic_reduction:
        beta = np.tanh(w * np.sqrt(gamma)) * alpha * np.sqrt(gamma)
        u[-1, :] = 1 / (3 + 2 * (z[1] - z[0]) * beta) * (4 * u[-2, :] - u[-3, :])

    tau = np.arange(0, 1, dtau)

    n_traces = 10
    colors = plt.get_cmap("viridis")(np.linspace(0.8, 0, n_traces))

    title = f"{eta=} {gamma=} {alpha=} {w=} {M=} {epsilon=} {dtau=}"

    for i in range(n_traces):
        j = int(i * (len(tau) / n_traces))
        plt.plot(z, u[:, j], label=f"t={tau[j]:.2f}", color=colors[i])
    plt.legend(loc="lower left", fontsize=6)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1 + w)
    plt.grid()
    plt.title(title, fontsize=9.5)
    
    try:
        plt.savefig(f"figures/{title}.png")
    except:
        print("Couldn't save file since 'figures' directory doesn't exist")
    plt.show()
    plt.clf()

    if surface_plot == True:
        plot3d(z, tau, u)


if __name__ == "__main__":
    for M in [100]:
        main(M=M, analytic_reduction=True)

# %%
