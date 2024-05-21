import numpy as np
from core.classes import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Methods
    -------
    __init__() -> None
        Initializes the Sigmoid activation, setting up the cache.
    forward(x: np.ndarray) -> np.ndarray
        Applies the Sigmoid function to the input array and caches the result.
    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the Sigmoid function with respect to the input.
    __call__(x: np.ndarray) -> np.ndarray
        Enables the instance to be called as a function, applying the forward pass.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.cache['y'] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'].append(x)
        y = 1 / (1 + np.exp(-x))
        self.cache['y'].append(y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        y = self.cache['y'].pop()
        return grad * y * (1 - y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Tanh(Activation):
    """
    Tanh activation function.

    Methods
    -------
    __init__() -> None
        Initializes the Tanh activation, setting up the cache.
    forward(x: np.ndarray) -> np.ndarray
        Applies the Tanh function to the input array and caches the result.
    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the Tanh function with respect to the input.
    __call__(x: np.ndarray) -> np.ndarray
        Enables the instance to be called as a function, applying the forward pass.
    """
    def __init__(self) -> None:
        super().__init__()
        self.cache['y'] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'].append(x)
        y = np.tanh(x)
        self.cache['y'].append(y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.cache['x'].pop()
        return grad * (1 - np.square(self.cache['y'].pop()))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) activation function.

    Methods
    -------
    __init__() -> None
        Initializes the ReLU activation, setting up the cache.
    forward(x: np.ndarray) -> np.ndarray
        Applies the ReLU function to the input array.
    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the ReLU function with respect to the input.
    __call__(x: np.ndarray) -> np.ndarray
        Enables the instance to be called as a function, applying the forward pass.
    """
            
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'].append(x)
        y = np.maximum(0, x)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache['x'].pop()
        dx = np.ones_like(x)
        dx[x < 0] = 0
        return grad * dx

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Softmax(Activation):
    """
    Softmax activation function.

    Methods
    -------
    __init__() -> None
        Initializes the Softmax activation, setting up the cache.
    forward(x: np.ndarray) -> np.ndarray
        Applies the Softmax function to the input array and caches the result.
    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the Softmax function with respect to the input.
    __call__(x: np.ndarray) -> np.ndarray
        Enables the instance to be called as a function, applying the forward pass.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        exp_x = np.exp(x - np.max(x))
        softmax_output = exp_x / np.sum(exp_x + 1e-12, axis=1, keepdims=True)
        self.cache['y'] = softmax_output
        return softmax_output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        softmax_output = self.cache['y']
        jacobian_matrix = np.zeros((softmax_output.shape[0], softmax_output.shape[1], softmax_output.shape[1]))
        for i in range(softmax_output.shape[0]):
            for j in range(softmax_output.shape[1]):
                for k in range(softmax_output.shape[1]):
                    if j == k:
                        jacobian_matrix[i, j, k] = softmax_output[i, j] * (1 - softmax_output[i, k])
                    else:
                        jacobian_matrix[i, j, k] = -softmax_output[i, j] * softmax_output[i, k]
        grad_input = np.matmul(grad.reshape(grad.shape[0], 1, grad.shape[1]), jacobian_matrix).squeeze()
        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class LogSoftmax(Activation):
    """
    LogSoftmax activation function.

    Methods
    -------
    __init__() -> None
        Initializes the LogSoftmax activation, setting up the cache.
    forward(x: np.ndarray) -> np.ndarray
        Applies the LogSoftmax function to the input array and caches the result.
    backward(grad: np.ndarray) -> np.ndarray
        Computes the gradient of the LogSoftmax function with respect to the input.
    __call__(x: np.ndarray) -> np.ndarray
        Enables the instance to be called as a function, applying the forward pass.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract the maximum value for numerical stability
        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max

        # Compute the log-softmax
        exp_x = np.exp(x_shifted)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        log_softmax = x_shifted - np.log(sum_exp_x)

        self.cache['x'] = x
        self.cache['exp_x'] = exp_x
        self.cache['sum_exp_x'] = sum_exp_x

        return log_softmax

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x, exp_x, sum_exp_x = self.cache['x'], self.cache['exp_x'], self.cache['sum_exp_x']

        # Compute the gradient of log-softmax
        dx = exp_x / sum_exp_x
        dx = (dx - np.exp(x - x.max(axis=1, keepdims=True))) * grad

        return dx

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
