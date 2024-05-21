from typing import TypedDict, List

import numpy as np

class Autograd:

    def __init__(self) -> None:
        self.cache: dict = {}

    def forward(self, *args) -> None:
        raise NotImplementedError()

    def backward(self, *args) -> None:
        raise NotImplementedError()

    def __call__(self, *args) -> None:
        raise NotImplementedError()

class Weights(TypedDict):
    w: np.ndarray
    b: np.ndarray
    dw: np.ndarray
    db: np.ndarray

class Layer(Autograd):
    weights: Weights = {}

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args) -> None:
        raise NotImplementedError()

    def backward(self, *args) -> None:
        raise NotImplementedError()

    def init_weights(self, *args) -> None:
        raise NotImplementedError()

class Loss(Autograd):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.value: np.float32 = 0.0
        self.regularization_coefficient: float = 0.0

    def forward(self, *args) -> None:
        raise NotImplementedError()

    def grad(self, *args) -> None:
        raise NotImplementedError()

    def regularize_fn(self) -> None:
        if self.regularize == "l1":
            for layer in self.model.parameters:
                self.regularization_coefficient += np.sum(np.abs(layer["w"]))
        elif self.regularize == "l2":
            for layer in self.model.parameters:
                self.regularization_coefficient += np.sum(np.square(layer["w"]))
        self.value += self.alpha * self.regularization_coefficient
        self.regularization_coefficient = 0.0

    def backward(self) -> None:
        y_hat = self.grad()
        y_hat = self.model.final_activation.backward(y_hat)
        L = len(self.model.layers)
        for i, layer in enumerate(self.model.layers[::-1]):
            layer.backward(y_hat)
            if self.regularize == "l2":
                layer.weights['dw']+= self.alpha * layer.weights['w']
            elif self.regularize == "l1":
                layer.weights['dw']+= self.alpha * np.sign(layer.weights['w'] + 1e-8)
            if L - i - 1 >= 1:
                l__h_prev = np.dot(y_hat, layer.weights['w'])
                y_hat = self.model.activation.backward(l__h_prev)

    def __call__(self, *args) -> None:
        self.forward(*args)
        return self
    
class Activation(Autograd):

    def __init__(self) -> None:
        super().__init__()
        self.cache['x'] = []

    def forward(self, *args) -> None:
        raise NotImplementedError()

    def backward(self, *args) -> None:
        raise NotImplementedError()

class Optimizer:
    def __init__(self) -> None:
        self.params = {}

    def step(self, *args) -> None:
        raise NotImplementedError()

    def zero_grad(self, *args) -> None:
        raise NotImplementedError()

class Module:
    
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        y = self.layers[-1](x)
        o = self.final_activation(y)
        return o
    
    def __call__(self, x) -> np.ndarray:
        return self.forward(x)
    
    @property
    def parameters(self) -> List[Weights]:
        return [layer.weights for layer in self.layers]