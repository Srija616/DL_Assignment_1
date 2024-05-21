from core.classes import Loss, Module
import numpy as np

class MSE(Loss):
    def __init__(self, model: Module, regularize: str="none", alpha: float=5e-3) -> None:
        super().__init__(model)
        self.regularize = regularize
        self.alpha = alpha

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        # self.value = np.squeeze(np.mean(np.square(y_pred - y_true)))
        self.value = np.squeeze(np.sum(np.square(y_pred - y_true)) / y_pred.shape[0])
        self.regularize_fn()
        return self
    
    def grad(self) -> np.ndarray:
        return 2 * (self.cache['y_pred'] - self.cache['y_true'])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        return self.forward(y_pred, y_true)

class CrossEntropy(Loss):
    def __init__(self, model: Module, regularize: str="none", alpha: float=5e-3) -> None:
        super().__init__(model)
        self.regularize = regularize
        self.alpha = alpha

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.value = np.squeeze(-np.sum(y_true * np.log(y_pred + 1e-12)))
        self.regularize_fn()
        return self

    def grad(self) -> np.ndarray:
        return self.cache['y_pred'] - self.cache['y_true']

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        return self.forward(y_pred, y_true)

class NLLLoss(Loss):
    def __init__(self, model, regularize: str="none", alpha: float=5e-3):
        super().__init__(model)
        # self.model = model
        self.regularize = regularize
        self.alpha = alpha

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        self.cache = {}
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true

        log_probs = y_pred[np.arange(len(y_true)), np.argmax(y_true, axis=1)]
        self.value = -np.mean(log_probs)

        self.regularize_fn()

        return self

    def grad(self) -> np.ndarray:
        y_pred, y_true = self.cache['y_pred'], self.cache['y_true']
        batch_size = y_pred.shape[0]
        y_pred_grad = np.zeros_like(y_pred)
        y_pred_grad[np.arange(batch_size), np.argmax(y_true, axis=1)] = -1 / batch_size

        return y_pred_grad

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float32:
        return self.forward(y_pred, y_true)