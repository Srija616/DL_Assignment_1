from core.classes import Optimizer
import numpy as np

class GradientDescent(Optimizer):
    def __init__(self, parameters: list, learning_rate: float=1e-3) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self, *args) -> None:
        for i, layer in enumerate(self.parameters):
            layer['w'] -= self.learning_rate * layer['dw']
            layer['b'] -= self.learning_rate * layer['db']
    
    def zero_grad(self, *args) -> None:
        for layer in self.parameters:
            layer['dw'] = np.zeros_like(layer['dw'])
            layer['db'] = np.zeros_like(layer['db'])

class SGDM(Optimizer):
    def __init__(self, parameters: list, learning_rate: float=1e-3, momentum: float=0.9, epsilon: float=1e-12) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"vdB{i}"] = np.zeros_like(layer['db'])
    
    def step(self, n_iter: int) -> None:
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = self.momentum * self.params[f"vdW{i}"] + self.learning_rate * layer['dw']
            self.params[f"vdB{i}"] = self.momentum * self.params[f"vdB{i}"] + self.learning_rate * layer['db']
            layer['w'] -= self.learning_rate * self.params[f"vdW{i}"]
            layer['b'] -= self.learning_rate * self.params[f"vdB{i}"]

    def zero_grad(self, *args) -> None:
        for layer in self.parameters:
            layer['dw'] = np.zeros_like(layer['dw'])
            layer['db'] = np.zeros_like(layer['db'])

class Nesterov(Optimizer):
    def __init__(self, parameters: list, learning_rate: float=1e-3, momentum: float=0.9, epsilon: float=1e-12) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"vdB{i}"] = np.zeros_like(layer['db'])
    
    def step(self, n_iter: int) -> None:
        for i, layer in enumerate(self.parameters, start=1):
            vdw_prev = self.params[f"vdW{i}"]
            vdb_prev = self.params[f"vdB{i}"]
            self.params[f"vdW{i}"] = self.momentum * self.params[f"vdW{i}"] - self.learning_rate * layer['dw']
            self.params[f"vdB{i}"] = self.momentum * self.params[f"vdB{i}"] - self.learning_rate * layer['db']
            layer['w'] += self.learning_rate * ( self.momentum * vdw_prev + (1 - self.momentum) * self.params[f"vdW{i}"])
            layer['b'] += self.learning_rate * ( self.momentum * vdb_prev + (1 - self.momentum) * self.params[f"vdB{i}"])

    def zero_grad(self, *args) -> None:
        for layer in self.parameters:
            layer['dw'] = np.zeros_like(layer['dw'])
            layer['db'] = np.zeros_like(layer['db'])

class RMSProp(Optimizer):
    def __init__(self, parameters: list, learning_rate: float=1e-3, momentum: float=0.9, epsilon: float=1e-12) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"sdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"sdB{i}"] = np.zeros_like(layer['db'])
    
    def step(self, n_iter: int) -> None:
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"sdW{i}"] = self.momentum * self.params[f"sdW{i}"] + (1 - self.momentum) * layer['dw'] ** 2
            self.params[f"sdB{i}"] = self.momentum * self.params[f"sdB{i}"] + (1 - self.momentum) * layer['db'] ** 2
            # self.params[f"sdW{i}"] /= (1 - self.momentum ** n_iter + self.epsilon)
            # self.params[f"sdB{i}"] /= (1 - self.momentum ** n_iter + self.epsilon)
            layer['w'] -= self.learning_rate * layer['dw'] / (np.sqrt(self.params[f"sdW{i}"]) + self.epsilon)
            layer['b'] -= self.learning_rate * layer['db'] / (np.sqrt(self.params[f"sdB{i}"]) + self.epsilon)

    def zero_grad(self, *args) -> None:
        for layer in self.parameters:
            layer['dw'] = np.zeros_like(layer['dw'])
            layer['db'] = np.zeros_like(layer['db'])

class Adam(Optimizer):
    def __init__(self, parameters: list, learning_rate: float=1e-3, beta_1: float=0.9, beta_2: float=0.995, epsilon: float=1e-8) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"vdB{i}"] = np.zeros_like(layer['db'])
            self.params[f"sdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"sdB{i}"] = np.zeros_like(layer['db'])
    
    def step(self, n_iter: int) -> None:
        lr = self.learning_rate * np.sqrt(1 - self.beta_2 ** n_iter) / (1 - self.beta_1 ** n_iter + self.epsilon)
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = self.beta_1 * self.params[f"vdW{i}"] + (1 - self.beta_1) * layer['dw']
            self.params[f"vdB{i}"] = self.beta_1 * self.params[f"vdB{i}"] + (1 - self.beta_1) * layer['db']
            self.params[f"sdW{i}"] = self.beta_2 * self.params[f"sdW{i}"] + (1 - self.beta_2) * np.square(layer['dw'])
            self.params[f"sdB{i}"] = self.beta_2 * self.params[f"sdB{i}"] + (1 - self.beta_2) * np.square(layer['db'])
            layer['w'] -= (lr / (np.sqrt(self.params[f"sdW{i}"] + self.epsilon))) * self.params[f"vdW{i}"]
            layer['b'] -= (lr / (np.sqrt(self.params[f"sdB{i}"] + self.epsilon))) * self.params[f"vdB{i}"]
    
    def zero_grad(self, *args) -> None:
        for layer in self.parameters:
            layer['dw'] = np.zeros_like(layer['dw'])
            layer['db'] = np.zeros_like(layer['db'])

class Nadam(Optimizer):
    def __init__(self, parameters: list, learning_rate: float=1e-3, beta_1: float=0.9, beta_2: float=0.995, epsilon: float=1e-6) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"vdB{i}"] = np.zeros_like(layer['db'])
            self.params[f"sdW{i}"] = np.zeros_like(layer['dw'])
            self.params[f"sdB{i}"] = np.zeros_like(layer['db'])
    
    def step(self, n_iter: int) -> None:
        lr = self.learning_rate * np.sqrt(1 - self.beta_2 ** n_iter) / (1 - self.beta_1 ** n_iter + self.epsilon)
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = self.beta_1 * self.params[f"vdW{i}"] + (1 - self.beta_1) * layer['dw']
            self.params[f"vdB{i}"] = self.beta_1 * self.params[f"vdB{i}"] + (1 - self.beta_1) * layer['db']
            self.params[f"sdW{i}"] = self.beta_2 * self.params[f"sdW{i}"] + (1 - self.beta_2) * layer['dw'] ** 2
            self.params[f"sdB{i}"] = self.beta_2 * self.params[f"sdB{i}"] + (1 - self.beta_2) * layer['db'] ** 2
            layer['w'] -= lr * (self.beta_1 * self.params[f"vdW{i}"] + ((1 - self.beta_1) * layer['dw']) / (1 - self.beta_1 ** n_iter)) / (np.sqrt(self.params[f"sdW{i}"]) + self.epsilon)
            layer['b'] -= lr * (self.beta_1 * self.params[f"vdB{i}"] + ((1 - self.beta_1) * layer['db']) / (1 - self.beta_1 ** n_iter)) / (np.sqrt(self.params[f"sdB{i}"]) + self.epsilon)

    def zero_grad(self, *args) -> None:
        for layer in self.parameters:
            layer['dw'] = np.zeros_like(layer['dw'])
            layer['db'] = np.zeros_like(layer['db'])