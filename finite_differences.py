import matplotlib.pyplot as plt
import numpy as np

f = lambda x : np.sin(x)
class Solucionar_EDP(object):
    def __init__(self, L, T, N, dt, f):
        self.L = L
        self.T = T
        self.N = N
        self.dt = dt
        self.f = f
        self.soluciones = []
        return

    def calcul_MatA(self):
        N = self.N
        h = self.L / N
        alpha = self.dt / (2* h**3)
        A = np.zeros((N-1,N-1))
        values1 = [1 + 2*alpha, -6*alpha, 6*alpha, -2*alpha, 1 - 2*alpha]
        values2 = [alpha, -2*alpha, 1, 2*alpha, -alpha]
        for i in range(N-1):
            if i == 0:
                A[i][0] = values1[0]
                A[i][1] = values1[1]
                A[i][2] = values1[2]
                A[i][3] = values1[3]
                continue
            if i == N - 2:
                A[i][N-5] = -values1[3]
                A[i][N-4] = -values1[2]
                A[i][N-3] = -values1[1]
                A[i][N-2] = -values1[-1]
                continue
            if i > 0 and i < N - 2:
                for j in range(N-1):
                    if i == 1:
                        if j <= 3:
                            A[i][j] = values2[j+1]
                            continue
                        else:
                            break
                    if i == (N - 3):
                        if j >= (N - 5):
                            A[i][j] = values2[j - N + 5]
                            continue
                        else:
                            continue
                    if j + (2 - i) < len(values2) and j + (2-i) >= 0:
                        A[i][j] = values2[j + (2 - i)]
        return A

    def calcul_MatB(self, u_n):
        N = self.N
        diag_up = [-u_n[i] for i in range(0,len(u_n)-1)]
        diag_down = [u_n[i] for i in range(1,len(u_n))]
        B = np.zeros((N-1,N-1))
        B[0][1] = diag_up[0]
        B[-1][-2] = diag_down[-1]
        for i in range(1,N-2):
            B[i][i+1] = diag_up[i]
            B[i][i-1] = diag_down[i] 
        return B

    def auxvect(self):
        N = self.N
        L = self.L
        x = np.linspace(0, L, N+1)
        F = [self.f(x[i]) for i in range(1,N)]
        return F
    
    def joinMat(self, u_n):
        A = self.calcul_MatA()
        B = self.calcul_MatB(u_n)
        N = self.N
        h = self.L / N
        beta = self.dt / (2*h)
        return A + beta*B
    
    def solinicial(self):
        u0 = lambda x : 0
        #u0 = lambda x : np.cos(np.pi * x)
        return [u0(i) for i in range(0,N-1)]

    def recursion_u(self):
        t = 0
        u_n = np.array(self.solinicial())
        mat = np.array(self.joinMat(u_n))
        vect = np.array(self.auxvect())
        self.soluciones.append((t, u_n))
        prox_u = mat @ u_n + self.dt * vect
        t += dt
        self.soluciones.append((t,prox_u))
        while(t < self.T):
            u_n = self.soluciones[-1][1]
            mat = self.joinMat(u_n)
            prox_u = mat @ u_n + self.dt * vect
            t += dt
            self.soluciones.append((t,prox_u))
        return self.soluciones

    def kdv_finite_differences(self):
        sol = self.recursion_u()
        #x = np.linspace(0, self.L, self.N + 1)
        x = [(2*np.pi / self.N)*i for i in range(-N//2 - 1, N//2)]
        t = 0
        while(t < self.T / self.dt + 1):
            u_t = np.concatenate(([0], sol[t][1], [0]))
            plt.plot(x, u_t)
            plt.title(f"Gráfico dinámico para la solución de la ecuación KdV con dt: {self.dt}", pad=18)
            plt.xlabel(f"Eje x - Tiempo: {round(t*self.dt, 3)} ticks")
            plt.ylabel("Valores de la función solución obtenida")
            plt.xlim(-2, 2)
            plt.ylim(-0.075,0.2)
            plt.grid()
            plt.pause(1e-14)
            if (t != self.T / self.dt) : plt.clf()
            t+=1
        plt.show() 
        return

L = 50
T = 5
N = 100
dt = 0.01
f = lambda x : 0.1*(1/(np.cosh(x-25))**2)
#f = lambda x : 0
sol = Solucionar_EDP(L, T, N, dt, f)
sol.kdv_finite_differences()
