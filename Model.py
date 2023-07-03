import numpy as np
import time

# BAsed on Lecture Slides

class GD_fix_gamma():
    def __init__(self, max_iters=100,learning_rate=0.1, alpha=0):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.alpha = alpha

    def set_parameter(self, param):
        self.learning_rate=param[0]
        self.alpha=param[1]

    def get_name(self):
        return "GD"

    def get_parameter(self):
        return {
            "max_iters": self.max_iters,
            "lr": self.learning_rate,
            "alpha": self.alpha,
        }

    def fit(self, X, Y):
        b = Y
        A = np.c_[np.ones(len(Y)), X]
        gamma = self.learning_rate
        x_start = np.zeros(A.shape[1])
        return self.fitting(b,A,gamma,x_start)

    def fitting(self,b,A,gamma,x_start):
        start_time = time.time()
        train_error, parameter_x = self.gradient_descent(b, A, x_start, self.max_iters, gamma)
        end_time = time.time()
        fit_runtime = end_time - start_time
        self.x_final = parameter_x
        return fit_runtime, train_error

    def calculate_objective(self,Axmb):
        """Calculate the mean squared error for vector Axmb = Ax - b."""
        return 0.5*np.mean(Axmb ** 2)

    def calculate_L(self,b, A):
        """Calculate the smoothness constant for f"""
        norm = np.linalg.norm(np.dot(A.T, A), 2)
        L = norm / b.shape[0]
        return L

    def compute_gradient(self,b, A, x):
        """Compute the gradient."""
        Axmb = A.dot(x) - b
        grad = A.T.dot(Axmb) / len(Axmb)
        grad_lasso = self.alpha * np.sign(x)
        grad+=grad_lasso
        mse=self.calculate_objective(Axmb)
        return grad, mse

    def gradient_descent(self,b, A, initial_x, max_iters, gamma):
        """Gradient descent algorithm."""
        # Define parameters to store x and objective func. values
        xs = [initial_x]
        mses = []
        x = initial_x
        works=True
        for n_iter in range(max_iters):
            grad, mse = self.compute_gradient(b, A, x)
            mses.append(mse)

            x = x - gamma * grad
            if np.isnan(x).any() or np.isinf(x).any():
                works=False
                break
            #print(x)
            xs.append(x)
        mses.append(self.calculate_objective(A.dot(x) - b))
        return mses, xs

    def predict(self, Xdata):
        num_samples = len(Xdata)
        tx = np.c_[np.ones(num_samples), Xdata]
        return [tx.dot(x) for x in self.x_final]

class GD(GD_fix_gamma):
    def __init__(self,max_iters,learning_rate=0.1, alpha=0):
        super().__init__(max_iters=max_iters,learning_rate=learning_rate, alpha=alpha)

    def get_parameter(self):
        return {
            "max_iters": self.max_iters,
            "alpha": self.alpha,
        }

    def get_name(self):
        return "GD gamma=1/L"

    def fit(self, X, Y):
        b = Y
        A = np.c_[np.ones(len(Y)), X]
        gamma = 1 / self.calculate_L(b, A)
        x_start = np.zeros(A.shape[1])
        return self.fitting(b,A,gamma,x_start)

class SGD(GD):
    def __init__(self,max_iters,learning_rate=0.1, batch_size=16, alpha=0,decreasing_learning_rate=False):
        super().__init__(max_iters=max_iters,learning_rate=learning_rate, alpha=alpha)
        self.batch_size=batch_size
        self.decreasing_learning_rate=decreasing_learning_rate

    def get_name(self):
        return "SGD_"+str(self.batch_size)

    def get_parameter(self):
        return {
            "max_iters": self.max_iters,
            "lr": self.learning_rate,
            "alpha": self.alpha,
            "decreasing_learning_rate": self.decreasing_learning_rate
        }

    def fitting(self,b,A,gamma,x_start):
        start_time = time.time()
        train_error, parameter_x = self.stochastic_gradient_descent(b, A, x_start ,self.max_iters, gamma,self.batch_size)
        end_time = time.time()
        fit_runtime = end_time - start_time
        self.x_final = parameter_x
        return fit_runtime, train_error


    def stochastic_gradient_descent(self,b,A,initial_x,max_iters,gamma,batch_size):

        xs = [initial_x]  # parameters after each update
        mses = []
        x = initial_x

        for iteration in range(max_iters):
            grad = self.stochastic_gradient(b, A, x, batch_size=batch_size)
            if self.decreasing_learning_rate:
                lr = gamma / np.sqrt((iteration + 1))#(iteration + 1)
            else:
                lr = gamma

            mses.append(self.calculate_objective(A.dot(x) - b))
            # update x through the stochastic gradient update
            x = x - lr * grad

            # store x and objective
            xs.append(x.copy())

        mses.append(self.calculate_objective(A.dot(x) - b))
        return mses, xs


    def stochastic_gradient(self, b, A, x, batch_size=1):
        dataset_size = len(b)
        indices = np.random.choice(dataset_size, batch_size, replace=False)

        b=b[indices]
        A= A[indices, :]

        grad,mse=self.compute_gradient(b, A, x)
        return grad

class ADAM(SGD):
    def __init__(self, max_iters,learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0, batch_size=16,decreasing_learning_rate=False):
        super().__init__(max_iters=max_iters,learning_rate=learning_rate, alpha=alpha,batch_size=batch_size,decreasing_learning_rate=decreasing_learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_name(self):
        return "ADAM_"+str(self.batch_size)

    def get_parameter(self):
        return {
            "max_iters": self.max_iters,
            "lr": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.epsilon,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "decreasing_learning_rate": self.decreasing_learning_rate
        }

    def fit(self, X, Y):
        b = Y
        A = np.c_[np.ones(len(Y)), X]
        gamma = self.learning_rate
        x_start = np.zeros(A.shape[1])
        return self.fitting(b, A, gamma, x_start)

    def fitting(self, b, A, gamma, x_start):
        m = np.zeros_like(x_start)
        v = np.zeros_like(x_start)
        start_time = time.time()
        train_error, parameter_x = self.adam_gradient_descent(b, A, x_start, self.max_iters, self.learning_rate,m,v,self.batch_size)
        end_time = time.time()
        fit_runtime = end_time - start_time
        self.x_final = parameter_x
        return fit_runtime, train_error


    def adam_gradient_descent(self, b, A, initial_x, max_iters, gamma, m ,v,batch_size):
        """Gradient descent algorithm."""
        # Define parameters to store x and objective func. values
        xs = [initial_x]
        mses = []
        x = initial_x
        for n_iter in range(max_iters):

            if self.decreasing_learning_rate:
                lr = gamma / np.sqrt((n_iter + 1))
            else:
                lr = gamma

            grad = self.stochastic_gradient(b, A, x,batch_size)
            mses.append(self.calculate_objective(A.dot(x) - b))
            t=n_iter+1
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)


            x = x - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            xs.append(x)

        mses.append(self.calculate_objective(A.dot(x) - b))
        return mses, xs

class RMSProp(SGD):
    def __init__(self, max_iters, learning_rate=0.001, beta=0.999, epsilon=1e-8,alpha=0, batch_size=16,decreasing_learning_rate=False):
        super().__init__(max_iters=max_iters,learning_rate=learning_rate, alpha=alpha,batch_size=batch_size,decreasing_learning_rate=decreasing_learning_rate)
        self.beta = beta
        self.epsilon = epsilon

    def get_name(self):
        return "RMSProp_"+str(self.batch_size)

    def get_parameter(self):
        return {
          "max_iters": self.max_iters,
          "lr": self.learning_rate,
          "beta": self.beta,
            "eps": self.epsilon,
            "alpha":self.alpha,
            "batch_size":self.batch_size,
            "decreasing_learning_rate": self.decreasing_learning_rate
        }

    def fit(self, X, Y):
        b = Y
        A = np.c_[np.ones(len(Y)), X]
        gamma = self.learning_rate
        x_start = np.zeros(A.shape[1])
        return self.fitting(b, A, gamma, x_start)

    def fitting(self, b, A, gamma, x_start):
        v = np.zeros_like(x_start)
        start_time = time.time()
        train_error, parameter_x = self.rms_gradient_descent(b, A, x_start, self.max_iters, self.learning_rate, v,self.batch_size)
        end_time = time.time()
        fit_runtime = end_time - start_time
        self.x_final = parameter_x
        return fit_runtime, train_error


    def rms_gradient_descent(self, b, A, initial_x, max_iters, gamma,  v,batch_size):
        """Gradient descent algorithm."""
        # Define parameters to store x and objective func. values
        xs = [initial_x]
        mses = []
        x = initial_x
        for n_iter in range(max_iters):

            if self.decreasing_learning_rate:
                lr = gamma / np.sqrt((n_iter + 1))
            else:
                lr = gamma
            grad = self.stochastic_gradient(b, A, x, batch_size)
            mses.append(self.calculate_objective(A.dot(x) - b))
            v = self.beta * v + (1 - self.beta) * (grad ** 2)

            x = x - lr * grad / (np.sqrt(v) + self.epsilon)

            xs.append(x)

        mses.append(self.calculate_objective(A.dot(x) - b))
        return mses, xs
