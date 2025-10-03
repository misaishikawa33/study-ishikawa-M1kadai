#5-3
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

def error(a, myu, sigma):
    rnd = np.random.normal(myu, sigma)
    return a + rnd

def normalizeCO(x, y, f):
    V = np.array([
        [x**2, x*y, 0, f*x, 0, 0],
        [x*y, x**2 + y**2, x*y, f*y, f*x, 0],
        [0, x*y, y**2, 0, f*y, 0],
        [f*x, f*y, 0, f**2, 0, 0],
        [0, f*x, f*y, 0, f**2, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    return 4 * V

def OLS(xi, N):
    M = np.zeros((6, 6))
    for i in range(len(xi)):
        A = xi[i].reshape(6, 1)
        M += np.dot(A, A.T)
    M = M / N
    value, vector = eigs(M, 1, which="SM")
    if vector.sum() < 0:
        vector = -vector
    return np.array(vector, dtype=float)

def MLE(xi, N, Vo):
    u = np.matrix(np.ones(6)).T
    M = np.zeros((6, 6), dtype=float)
    L = np.zeros((6, 6), dtype=float)
    count = 0
    threshold = 1e-1

    while True:
        M.fill(0)
        L.fill(0)
        for i in range(len(xi)):
            A = xi[i].reshape(6, 1)
            M_1 = np.dot(A, A.T).astype(float)
            M_2 = np.dot(u.T, np.dot(Vo[i], u)).astype(float)
            M += M_1 / M_2

            L_11 = np.dot(xi[i], u)**2
            L_11 = L_11[0, 0] * Vo[i]
            L_11 = L_11.astype(float)
            L += L_11 / (M_2**2)

        J = (M - L) / N
        u_old = u
        value, vector = eigs(J, 1, which="SM")
        u = np.matrix(vector)
        if u.sum() < 0:
            u = -u

        if np.linalg.norm(np.abs(u) - np.abs(u_old)) < threshold:
            break

        count += 1
        if count > 100:
            print("Maximum iterations reached.")
            break

    return np.array(vector, dtype=float)

def RMS(data, Pu):
    rsa = np.dot(Pu, data)
    RSA = np.dot(rsa.T, rsa)
    return RSA

def KCR(xi, Vo, u):
    A = xi.reshape(6, 1)
    M_1 = np.dot(A, A.T)
    M_2 = np.dot(u.T, np.dot(Vo, u))
    M = M_1 / M_2
    return M

def main(N, out_name):
    pi = math.pi
    myu = 0
    f = 1
    sigma_max = 20

    true = np.array([1 / 300**2, 0, 1 / 200**2, 0, 0, -1])
    true = true / np.linalg.norm(true)
    Pu = np.eye(6) - np.outer(true, true)

    Ols = []
    Mle = []
    X = []
    Y = []
    xaxis = np.linspace(0.1, 2, sigma_max - 1)
    D_kcr = []

    for i in range(N):
        theta = -(pi / 4) + ((11 * pi) / (12 * N) * i)
        x = 300 * math.cos(theta)
        y = 200 * math.sin(theta)
        X.append(x)
        Y.append(y)

    loop = 1000

    for sigma_idx in range(1, sigma_max):
        sigma = sigma_idx * 0.1
        print(f"********************{sigma}**********************")
        ols = 0
        mle = 0

        for _ in range(loop):
            xi = []
            Vo = []
            for j in range(N):
                X_e = error(X[j], myu, sigma)
                Y_e = error(Y[j], myu, sigma)
                xi.append(np.array([
                    X_e**2,
                    2 * X_e * Y_e,
                    Y_e**2,
                    2 * X_e * f,
                    2 * Y_e * f,
                    f**2
                ]))
                Vo.append(normalizeCO(X_e, Y_e, f))

            u_ols = OLS(xi, N)
            u_mle = MLE(xi, N, Vo)
            ols += RMS(u_ols, Pu)
            mle += RMS(u_mle, Pu)

        ols_rms = np.sqrt(ols / loop)
        mle_rms = np.sqrt(mle / loop)
        Ols.append(ols_rms.reshape(1))
        Mle.append(mle_rms.reshape(1))

        M_bar = np.zeros((6, 6))
        for k in range(N):
            xi_bar = np.array([
                X[k]**2, 2 * X[k] * Y[k], Y[k]**2,
                2 * X[k] * f, 2 * Y[k] * f, f**2
            ])
            Vo_bar = (sigma**2) * normalizeCO(X[k], Y[k], f)
            M_bar += KCR(xi_bar, Vo_bar, true)
        val, vec = np.linalg.eig(M_bar)
        val = np.sort(val)[::-1]
        kcr = sum(1 / val[j] for j in range(5))
        D_kcr.append(sigma * np.sqrt(kcr))

    print("ols:", Ols)
    print("************************")
    print("mle:", Mle)
    print("************************")
    print("kcr:", D_kcr)

    fig = plt.figure(1)
    plt.plot(xaxis, Ols, color="Red", label="OLS")
    plt.plot(xaxis, D_kcr, color="Green", label="KCR")
    plt.grid()
    plt.legend()
    plt.xlabel('standard deviation')
    plt.ylabel('RMS-error')
    plt.savefig(out_name)
    plt.show()

    fig = plt.figure(2)
    plt.plot(xaxis, Ols, color="Red", label="OLS")
    plt.plot(xaxis, Mle, color="Blue", label="MLE")
    plt.grid()
    plt.legend()
    plt.xlabel('standard deviation')
    plt.ylabel('RMS-error')
    plt.savefig("kadai4e-1.png")
    plt.show()

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print(f'Arguments are {len(args)}')
    else:
        N = int(sys.argv[1])
        out_name = str(sys.argv[2])
        main(N, out_name)
