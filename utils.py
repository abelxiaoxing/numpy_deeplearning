import numpy as np


def relu(x):
    f = np.maximum(0, x)
    df = np.where(x > 0, 1, 0)
    return f, df


def sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    df = f * (1 - f)
    return f, df


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    f = e_x / np.sum(e_x, axis=-1, keepdims=True)
    df = np.diagflat(f) - np.outer(f, f)
    return f, df


def tanh(x):
    f = np.tanh(x)
    df = 1 - f**2
    return f, df


def identity(x):
    f = x
    df = 1
    return f, df


def mish(x):
    exp_x = np.exp(np.clip(x, -100, 100))
    f = x * np.tanh(np.log(1 + exp_x))
    df = 1 + x * (1 - np.tanh(np.log(1 + exp_x)) ** 2) * exp_x / (1 + exp_x)
    return f, df

def silu(x):
    sigmoid = 1 / (1 + np.exp(-x))
    f = x * sigmoid
    df = sigmoid * (1 + x * (1 - sigmoid))
    return f, df

def selu(x):
    alpha = 1.67326
    lambd = 1.05070
    f = lambd * np.where(x > 0, x, alpha * np.exp(x) - alpha)
    df = lambd * np.where(x > 0, 1, alpha * np.exp(x))
    return f, df


def MSE(y_true, y_pred):
    loss = np.mean((y_pred - y_true) ** 2)
    gradient = 2 * (y_pred - y_true) / y_true.shape[0]
    # print(gradient[0])
    return loss, gradient


def SoftmaxCE(y_true, y_pred):
    y_pred = softmax(y_pred)[0]
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred))
    gradient = y_pred - y_true
    return loss, gradient


def SigmoidCE(y_true, y_pred):
    y_pred = sigmoid(y_pred)[0]
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    gradient = y_pred - y_true
    return loss, gradient

def save_weights(weights, filename="model_weights.npy"):
    with open(filename, 'wb') as f:
        np.save(f, weights)

def load_weights(filename="model_weights.npy"):
    with open(filename, 'rb') as f:
        return np.load(f, allow_pickle=True).item()

activation_table = {
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "tanh": tanh,
    "identity": identity,
    "mish": mish,
    "silu": silu,
    "selu": selu,
}

loss_table = {
    "MSE": MSE,
    "CE": SoftmaxCE,
}
