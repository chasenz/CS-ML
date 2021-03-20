import numpy as np

class SVM:
    def __init__(self, X, y, kernel_type="linear", max_iter=1000, C=1.0, tolerance=0.001):
        # parameters
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic
        }
        self.kernel_type = kernel_type

        self.max_passes = max_iter # max passes
        self.C = C # regularization paramtere
        self.tol = tolerance # tolerance

        # input/training-data
        self.X = X
        self.N, self.D = self.X.shape
        self.y = y


    def fit(self):
        alpha = np.zeros((self.N, ))
        b = 0
        passes = 0
        kernel = self.kernels[self.kernel_type]
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.N):
                Ei = self.get_Ek(i, self.get_w(alpha), b)
                if(((self.y[i]*Ei) < -self.tol and alpha[i] < self.C) or ((self.y[i]*Ei) > self.tol and alpha[i] > 0)):
                    j = self.get_rnd_int(self.N-1, i)  # Get random int i~=j
                    Ej = self.get_Ek(j, self.get_w(alpha), b)

                    alpha_old = np.copy(alpha) # save old alphas

                    L,H = self.get_L_H(alpha[j], alpha[i], self.y[j], self.y[i])

                    if(L == H):
                        continue
                    eta = -kernel(self.X[i], self.X[i]) -kernel(self.X[j], self.X[j]) + 2 * kernel(self.X[i], self.X[j])
                    if(eta>=0):
                        continue
                    alpha[j] = alpha[j] - self.y[j]*((Ei - Ej)/eta)
                    alpha[j] = max(alpha[j], L)
                    alpha[j] = min(alpha[j], H)
                    if(abs(alpha[j] - alpha_old[j]) < 1e-5):
                        continue

                    alpha[i] = alpha_old[i] + self.y[i]*self.y[j] * (alpha_old[j] - alpha[j])

                    b1 = b - Ei - self.y[i]*(alpha[i] - alpha_old[i])*np.dot(self.X[i], self.X[i].T) - self.y[j]*(alpha[j] - alpha_old[j])*np.dot(self.X[i], self.X[j].T)
                    b2 = b - Ej - self.y[i]*(alpha[i] - alpha_old[i])*np.dot(self.X[i], self.X[j].T) - self.y[j]*(alpha[j] - alpha_old[j])*np.dot(self.X[j], self.X[j].T)
                    if((0 < alpha[i] and alpha[i] < self.C) and (0 < alpha[j] and alpha[j] < self.C)):
                        b = (b1 + b2)/2
                    elif((0 < alpha[i] and alpha[i] < self.C)):
                        b = b1
                    elif ((0 < alpha[j] and alpha[j] < self.C)):
                        b = b2

                    num_changed_alphas+=1

            if(num_changed_alphas == 0):
                passes+=1
            else:
                passes = 0

        self.alpha = alpha
        self.w = self.get_w(alpha)
        self.b = b

        # print(passes)

        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = self.X[alpha_idx, :]
        return support_vectors, passes

    def predict(self, X):
        return np.sign(np.dot(self.w, X.T) + self.b).astype(int)

    def get_w(self, alpha):
        return np.dot(np.multiply(alpha, self.y), self.X)

    def get_rnd_int(self, n, z):
        # TODO: If does not work, use external method
        arr = np.arange(n)
        np.random.shuffle(arr)
        if arr[0] == z:
            return arr[1]
        else:
            return arr[0]

    def f(self, i, w, b):
        return np.sign(np.dot(w.T, self.X[i].T) + b).astype(int)

    def get_Ek(self, i, w, b):
        # print(np.dot(w.T, self.X[i].T) + b, self.y[i], self.X[i])
        return self.f(i,  w, b) - self.y[i]

    def get_L_H(self, alpha_j, alpha_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, alpha_j - alpha_i), min(self.C, self.C - alpha_i + alpha_j))
        else:
            return (max(0, alpha_i + alpha_j - self.C), min(self.C, alpha_i + alpha_j))

    #  Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)