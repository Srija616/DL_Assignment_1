import numpy as np
from core.classes import Layer

class Linear(Layer):
    """
    Methods
    -------
    __init__(input_size: int, output_size: int, init_strategy: str="he") -> None
        Initializes the layer with input size, output size, and weight initialization strategy.
    init_weights(strategy: str="he") -> None
        Initializes weights using the specified strategy.
    forward(x: np.ndarray) -> np.ndarray
        Computes the forward pass.
    backward(grad: np.ndarray) -> np.ndarray
        Computes the backward pass.
    __call__(x: np.ndarray) -> np.ndarray
        Calls the forward method.
    """
    def __init__(self, input_size: int, output_size: int, init_strategy: str="he") -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = {}
        self.init_weights(init_strategy)

    def init_weights(self, strategy:str="he") -> None:
        if strategy == "he":
            self.weights['w'] = np.random.normal(0, np.sqrt(2.0 / self.input_size), (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, np.sqrt(2.0 / self.input_size), (self.output_size, 1))
        elif strategy == "random":
            self.weights['w'] = np.random.normal(0, 1, (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, 1, (self.output_size, 1))
        elif strategy == "xavier":
            self.weights['w'] = np.random.normal(0, np.sqrt(6.0 / self.input_size), (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, np.sqrt(6.0 / self.input_size), (self.output_size, 1))
        elif strategy == "normal":
            self.weights['w'] = np.random.normal(0, 0.01, (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, 0.01, (self.output_size, 1))
        else:
            raise NotImplementedError()
        self.weights['dw'] = np.zeros_like(self.weights['w'])
        self.weights['db'] = np.zeros_like(self.weights['b'])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        return np.dot(x, self.weights['w'].T) + self.weights['b'].T

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.weights['dw'] = np.dot(self.cache['x'].T, grad).T
        self.weights['db'] = np.sum(grad, axis=0, keepdims=True).T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
class Dropout(Layer):
    """
    Dropout layer for regularization.

    Methods
    -------
    __init__(p: float=0.2) -> None
        Initializes the layer with dropout probability.
    forward(x: np.ndarray) -> np.ndarray
        Applies dropout to the input.
    backward(grad: np.ndarray) -> np.ndarray
        Applies the dropout mask to the gradient.
    __call__(x: np.ndarray) -> np.ndarray
        Calls the forward method.
    """
    def __init__(self, p: float=0.2) -> None:
        super().__init__()
        self.p = p
        self.cache['mask'] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        self.cache['mask'].append(np.random.binomial(1, 1 - self.p, size=x.shape))
        return x * self.cache['mask']

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.cache['mask'].pop()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)