import jax.numpy as jnp
from jax import grad
import numpy as np
import scipy.optimize as optimize

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def predict(w, b, X, gamma, lambd):
    return lambd + (1 - lambd - gamma) * sigmoid(jnp.dot(X, w) + b)

def loss(w, b, X, gamma, lambd, y):
    preds = predict(w, b, X, gamma, lambd)
    return -jnp.sum(jnp.log(preds * y + (1 - preds) * (1 - y)))

def sigmoid_SO(x):
        return 0.5 * (np.tanh(x / 2) + 1)

def predict_SO(w, b, X, gamma_R, gamma_L):
    return gamma_L + (1 - gamma_L - gamma_R) * sigmoid_SO(np.dot(X, w) + b)

def negloglik(params, X, y):
    w, b, gamma_R, gamma_L = params
    preds = predict_SO(w, b, X, gamma_R, gamma_L)
    llv = preds * y + (1 - preds) * (1 - y)
    return -np.sum(np.log(llv))


def fitSOGLM(X, y, w, b, gamma_R, gamma_L, method="SLSQP"):
    res = optimize.minimize(negloglik, 
	                     x0=[w, b, gamma_R, gamma_L], 
			     args=(X, y),
			     method=method)
    ests = {"w":res["x"][0], "b":res["x"][1], "gR":res["x"][2], "gL":res["x"][3]}
    return ests, res

    
def postPredict(est_obj, X, y):
    w_est, b_est, gamma_est, lambd_est = est_obj["x"]
    preds = predict_SO(w_est, b_est, X, gamma_est, lambd_est)
    acc = (y != preds.round()).mean()
    return preds, acc


def fitGDGLM(X, y, w, b, learning_rate, gamma, lambd, n_stop=int(5e3)):
    """X: jnp.array input data of shape (N, P)
    y: jnp.array binary choice data of shape (N, 1)
    w: jnp.array initialization for weights of shape (P, 1)
    b: scalar float, bias
    gamma: scalar float, chance level parameters
    lambd: scalar float, lapse parameter
    n_stop: greatest number of iterations fit can go on for
    """
    training_loss = np.zeros(n_stop)

    for k in range(n_stop):
        w -= learning_rate * grad(loss, argnums=0)(w, b, X, gamma, lambd, y)
        b -= learning_rate * grad(loss, argnums=1)(w, b, X, gamma, lambd, y)
        gamma -= learning_rate * grad(loss, argnums=3)(w, b, X, gamma, lambd, y)
        lambd -= learning_rate * grad(loss, argnums=4)(w, b, X, gamma, lambd, y)

        # save loss
        training_loss[k] = (y != predict(w, b, X, gamma, lambd).round()).mean()

        # check if GD should stop
        if k > 500 and np.var(training_loss[k-300:k]) == 0:
            break

    estimates = {"w":w, "b":b, "gamma":gamma, "lambd":lambd}

    return training_loss[:k], estimates
