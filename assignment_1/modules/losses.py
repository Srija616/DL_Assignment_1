from core.classes import Loss, Module
import numpy as np

class MSE(Loss):
    """
    Mean Squared Error (MSE) loss function.

    Methods
    -------
    __init__(model: Module, regularize: str="none", alpha: float=5e-3) -> None
        Initializes the MSE loss with a model, regularization type, and regularization strength.
    forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.float32
        Computes the MSE loss and applies regularization.
    grad() -> np.ndarray
        Computes the gradient of the MSE loss with respect to the predictions.
    __call__(y_pred: np.ndarray, y_true: np.ndarray) -> np.float32
        Calls the forward method.
    """
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
    """
    Cross-Entropy loss function.

    Methods
    -------
    __init__(model: Module, regularize: str="none", alpha: float=5e-3) -> None
        Initializes the Cross-Entropy loss with a model, regularization type, and regularization strength.
    forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.float32
        Computes the Cross-Entropy loss and applies regularization.
    grad() -> np.ndarray
        Computes the gradient of the Cross-Entropy loss with respect to the predictions.
    __call__(y_pred: np.ndarray, y_true: np.ndarray) -> np.float32
        Calls the forward method.
    """
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
    """
    Negative Log-Likelihood (NLL) loss function.

    Methods
    -------
    __init__(model: Module, regularize: str="none", alpha: float=5e-3) -> None
        Initializes the NLL loss with a model, regularization type, and regularization strength.
    forward(y_pred: np.ndarray, y_true: np.ndarray) -> np.float32
        Computes the NLL loss and applies regularization.
    grad() -> np.ndarray
        Computes the gradient of the NLL loss with respect to the predictions.
    __call__(y_pred: np.ndarray, y_true: np.ndarray) -> np.float32
        Calls the forward method.
    """
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