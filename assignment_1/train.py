import os
import numpy as np
import wandb
import json
from functools import partial
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar10, fashion_mnist
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

from core.classes import Module

from modules.layers import Linear
from modules.dataloader import DataLoader
from modules.activations import ReLU, Sigmoid, Tanh, LogSoftmax, Softmax
from modules.losses import NLLLoss, MSE, CrossEntropy
from modules.optimizers import GradientDescent, Adam, Nadam, RMSProp, SGDM, Nesterov

class DenseNetwork(Module):
    def __init__(self, n_features: int, n_classes: int, hidden_sizes: int, num_layers: int, hidden_activation: str, init_strategy: str="he") -> None:
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(Linear(n_features, hidden_sizes, init_strategy))
            else:
                self.layers.append(Linear(hidden_sizes, hidden_sizes, init_strategy))
        self.layers.append(Linear(hidden_sizes, n_classes, init_strategy))
        self.activation = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
        }.get(hidden_activation, ReLU())
        self.final_activation = Softmax()

# class HyperParameters(TypedDict):
#     hidden_activations: str
#     init_strategy: str
#     loss_fn: str
#     optimizer: str
#     learning_rate: float
#     n_epochs: int
#     batch_size: int
#     alpha: float
#     beta_1: float
#     beta_2: float
#     regularizer: str
#     hidden_sizes: List[int]
#     epsilon: float

default_hyperparameters = {
    'hidden_activations': 'tanh',
    'init_strategy': 'he',
    'loss_fn': 'cross_entropy',
    'optimizer': 'nadam',
    'learning_rate': 1e-4,
    'n_epochs': 10,
    'batch_size': 16,
    'alpha': 1e-4,
    'beta_1': 0.9,
    'beta_2': 0.995,
    'regularizer': 'l1',
    'hidden_sizes': 64,
    'num_layers': 4,
    'epsilon': 1e-8,
}

def main():
    parser = ArgumentParser()

    parser.add_argument("-wp", "--wandb-project", type=str, default="cs6910-assignment1", help="Wandb project to use for logging")
    parser.add_argument("-we", "--wandb-entity", type=str, default="Srija17199", help="Wandb entity to use for logging")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", help="Dataset to use for training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning Rate for Optimizers")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Use as momentum for SGDM, RMSProp, and Nesterov")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.995, help="Beta 2 for Adam and Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Numerical Stability Constant")
    parser.add_argument("-a", "--alpha", type=float, default=1e-4, help="Use as weight decay (regularization coefficient)")
    parser.add_argument("-w_i", "--weight_init", type=str, default="he", help="Weight initialization strategy")
    parser.add_argument("-sz", "--hidden_sizes", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("-ac", "--activation", type=str, default="relu", help="Activation function for hidden layers")
    parser.add_argument("-r", "--regularizer", type=str, default="l1", help="Regularizer to use")
    parser.add_argument("--sweep", action="store_true", help="Whether to run sweep", default = False)
    parser.add_argument("--conf_path", type=str, help="Path to config file", default = "./sweep_config.json")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb for logging")

    args = parser.parse_args()
    hyperparameters = {
        'hidden_activations': args.activation,
        'init_strategy': args.weight_init,
        'loss_fn': args.loss,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'n_epochs': args.epochs,
        'batch_size': args.batch_size,
        'alpha': args.alpha,
        'beta_1': args.beta1,
        'beta_2': args.beta2,
        'regularizer': args.regularizer,
        'hidden_sizes': args.hidden_sizes,
        'epsilon': args.epsilon,
        'num_layers': args.num_layers,
        }

    # wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=hyperparameters)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    train_loader, val_loader, test_loader = get_data(args.dataset, args.batch_size)
    n_features = train_loader.x.shape[1]
    n_classes = train_loader.y.shape[1]
    if args.sweep:
        assert os.path.exists(args.conf_path)
        perform_sweep(args.conf_path, args.wandb_project, args.wandb_entity, hyperparameters, train_loader, val_loader, n_features, n_classes)
    else:
        model = build_model(n_features, n_classes, hyperparameters)
        loss_fn = build_loss_model(model, n_classes, hyperparameters)
        optimizer = build_optimizer(model, hyperparameters, hyperparameters["optimizer"])
        train(hyperparameters, model, loss_fn, optimizer, train_loader, val_loader, test_loader)

    

def build_model(n_features, n_classes, hyperparameters):
    return DenseNetwork(
        n_features, 
        n_classes, 
        hyperparameters["hidden_sizes"],
        hyperparameters["num_layers"],
        hyperparameters["hidden_activations"], 
        hyperparameters['init_strategy']
        )

def build_loss_model(model, loss_fn, hyperparameters):
    loss_fn = {
            'cross_entropy': CrossEntropy,
            'mse': MSE,
        }.get(loss_fn, NLLLoss)
    return loss_fn(model, hyperparameters['regularizer'], hyperparameters['alpha'])

def build_optimizer(model, hyperparameters, optimizer):
    optimizer = {
        'sgd': GradientDescent(model.parameters, hyperparameters['learning_rate']),
        'momentum': SGDM(model.parameters, hyperparameters['learning_rate'], hyperparameters['beta_1'], hyperparameters["epsilon"]),
        'nag': Nesterov(model.parameters, hyperparameters['learning_rate'], hyperparameters['beta_1'], hyperparameters["epsilon"]),
        'rmsprop': RMSProp(model.parameters, hyperparameters['learning_rate'], hyperparameters['beta_1'], hyperparameters["epsilon"]),
        'adam': Adam(model.parameters, hyperparameters['learning_rate'], hyperparameters['beta_1'], hyperparameters['beta_2'], hyperparameters["epsilon"]),
        'nadam': Nadam(model.parameters, hyperparameters['learning_rate'], hyperparameters['beta_1'], hyperparameters['beta_2'], hyperparameters["epsilon"]),
    }.get(optimizer, GradientDescent(model.parameters, hyperparameters['learning_rate']))
    return optimizer

def get_data(dataset: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        class_labels = [str(i) for i in range(10)]
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise NotImplementedError("Dataset not implemented")
    
    plot_images(x_train, y_train, class_labels)
    train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    
def plot_images(data, labels, class_names=None, flatten=False):
    uniq_labels = np.unique(labels)

    fig, ax = plt.subplots(2,5, figsize=(15, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.reshape(-1)

    for i, label in enumerate(uniq_labels):
        img = data[np.where(labels == label)[0][0]]
        if class_names:
            ax[i].set_title(class_names[label])
        if flatten:
            img = img.reshape(28, 28)
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')

    wandb.log({"Class Images": fig})
    plt.close(fig)


def train(hyperparameters, model, loss_fn, optimizer, train_loader, val_loader, test_loader=None) -> None:
    val_loss, val_acc = 0, 0
    for epoch in range(hyperparameters["n_epochs"]):
        train_loss, train_acc = train_epoch(model, loss_fn, train_loader, optimizer)
        val_loss, val_acc = val_epoch(model, loss_fn, val_loader)
        # if not self.use_wandb:
        print(f"[{epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.4f}")
        # breakpoint()
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }, step=epoch+1)
    if test_loader:
        test_epoch(model, loss_fn, test_loader, ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

def train_epoch(model, loss_fn, train_loader, optimizer) -> None:
    train_loss = []
    train_acc = []
    for i, (images, labels) in enumerate(train_loader, start=1):
        # print(model.layers[-1].weights['w'])
        # self.plot_samples(images, labels)
        preds = model(images)
        # print(preds, labels)
        # print(preds.shape, labels.shape)
        loss_fn(preds, labels)
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step(i)
        train_loss.append(loss_fn.value)
        train_acc.extend(list(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)
    return train_loss, train_acc

def val_epoch(model, loss_fn, val_loader) -> None:
    val_loss = []
    val_acc = []
    for i, (images, labels) in enumerate(val_loader):
        preds = model(images)
        loss_fn(preds, labels)
        val_loss.append(loss_fn.value)
        val_acc.extend(list(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)
    return val_loss, val_acc

def test_epoch(model, loss_fn, test_loader, class_names) -> None:
    test_loss = []
    test_acc = []
    plot_preds, plot_labels = [], []
    for i, (images, labels) in enumerate(test_loader):
        preds = model(images)
        loss_fn(preds, labels)
        plot_preds.append(preds)
        plot_labels.append(labels)
        test_loss.append(loss_fn.value)
        test_acc.extend(np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
        print(preds.shape, labels.shape)
        # preds = np.concatenate(plot_preds)
        # labels = np.concatenate(plot_labels)
    cm = np.zeros((len(class_names), len(class_names)))

    for i in range(len(labels)):
        predicted_class = np.argmax(preds[i])  # Get the predicted class index
        true_class = np.argmax(labels[i])  # Get the true class index
        cm[true_class, predicted_class] += 1

    plot_confusion_matrix(cm, class_names)
    test_loss = np.mean(test_loss)
    test_acc = np.mean(test_acc)
    return test_loss, test_acc

def plot_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    """
    Creates and saves a beautiful confusion matrix plot.

    Args:
        cm: Confusion matrix (2D NumPy array).
        class_names: List of class names for the labels.
        filename: Filename to save the plot (default: "confusion_matrix.png").
    """

    plt.figure(figsize=(10, 7))  # Adjust figure size for clarity
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)  # Customize appearance
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")  # Rotate x-axis labels
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot figure


def sweep(hyperparameters, train_loader, val_loader, n_features, n_classes) -> None:
    config = wandb.config
    hparams = {**hyperparameters, **config}
    model = build_model(n_features, n_classes, hparams)
    optimizer = build_optimizer(model, hparams, hparams["optimizer"])
    loss_fn = build_loss_model(model,  hparams["loss_fn"], hparams)
    # breakpoint()
    # wandb.run_name = f'{len(hyperparameters["hidden_sizes"])}Layer-{hyperparameters["hidden_activations"]}Activated-{hyperparameters["optimizer"]}Optimized-{hyperparameters["loss_fn"]}Loss'
    train(hparams, model, loss_fn, optimizer, train_loader, val_loader)


# def predict(self, x) -> np.ndarray:
#     return np.argmax(self.model(x), axis=1)

def perform_sweep(conf_path, wandb_project_id, wandb_entity, hyperparameters, train_loader, val_loader, n_features, n_classes) -> None:
    with open(conf_path, 'r') as f:
        sweep_config = json.load(f)
    # sweep_id = wandb.sweep(sweep_config, project=wandb_project_id, entity=wandb_entity)
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_id, entity=wandb_entity)
    sweep_f = partial(sweep, hyperparameters, train_loader, val_loader, n_features, n_classes)
    wandb.agent(sweep_id, function=sweep_f)

# class Trainer:
#     def __init__(self, dataset: str, wandb_entity: str, wandb_project_id: str="CS6910-Assignment1", use_wandb: bool=True, do_sweep: bool=False, conf_path: str="sweep_config.json", **kwargs: HyperParameters) -> None:
#         self.wandb_entity = wandb_entity
#         self.use_wandb = use_wandb
#         self.wandb_project_id = wandb_project_id
#         self.do_sweep = do_sweep
#         self.conf_path = conf_path
#         self.hyperparameters = {**default_hyperparameters, **kwargs}
#         # if self.use_wandb:
#         #     wandb.init(project=self.wandb_project_id, entity=wandb_entity, config=self.hyperparameters)
#         self.train_loader, self.val_loader, self.test_loader = self.get_data(dataset, self.hyperparameters['batch_size'])
#         self.n_features = self.train_loader.x.shape[1]
#         self.n_classes = self.train_loader.y.shape[1]
#         self.set_model(self.hyperparameters['hidden_sizes'], self.hyperparameters['hidden_activations'])
#         self.set_loss_fn(self.hyperparameters['loss_fn'])
#         self.set_optimizer(self.hyperparameters['optimizer'])
        
#     def set_model(self, hidden_sizes: List[int], hidden_activation: str) -> None:
#         self.model = DenseNetwork(self.n_features, self.n_classes, hidden_sizes, hidden_activation, self.hyperparameters['init_strategy'])
    
#     def set_loss_fn(self, loss_fn: str) -> None:
#         self.loss_fn = {
#             'cross_entropy': NLLLoss(self.model, self.hyperparameters['regularizer'], self.hyperparameters['alpha']),
#             'mse': MSE(self.model, self.hyperparameters['regularizer'], self.hyperparameters['alpha']),
#         }.get(loss_fn, NLLLoss(self.model, self.hyperparameters['regularizer'], self.hyperparameters['alpha']))

#     def set_optimizer(self, optimizer: str) -> None:
#         self.optimizer = {
#             'sgd': GradientDescent(self.model.parameters, self.hyperparameters['learning_rate']),
#             'momentum': SGDM(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters["epsilon"]),
#             'nag': Nesterov(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters["epsilon"]),
#             'rmsprop': RMSProp(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters["epsilon"]),
#             'adam': Adam(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters['beta_2'], self.hyperparameters["epsilon"]),
#             'nadam': Nadam(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters['beta_2'], self.hyperparameters["epsilon"]),
#         }.get(optimizer, GradientDescent(self.model.parameters, self.hyperparameters['learning_rate']))
    
#     def get_data(self, dataset: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
#         if dataset == 'mnist':
#             (x_train, y_train), (x_test, y_test) = mnist.load_data()
#             x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
#             class_labels = [str(i) for i in range(10)]
#         elif dataset == 'cifar10':
#             (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#             x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
#             class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#         elif dataset == "fashion_mnist":
#             (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#             x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
#             class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#         else:
#             raise NotImplementedError("Dataset not implemented")
        
#         self.plot_images(x_train, y_train, class_labels, use_wandb=self.use_wandb)
#         train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(x_val, y_val, batch_size=batch_size, shuffle=False)
#         test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)
#         return train_loader, val_loader, test_loader
    
#     def train(self) -> None:
#         for epoch in range(self.hyperparameters["n_epochs"]):
#             self.train_epoch()
#             self.val_epoch()
#             if not self.use_wandb:
#                 print(f"[{epoch+1}] Train Loss: {self.train_loss:.4f} | Train Acc: {self.train_acc * 100:.4f} | Val Loss: {self.val_loss:.4f} | Val Acc: {self.val_acc * 100:.4f}")
#             if self.use_wandb:
#                 wandb.log({
#                     "train_loss": self.train_loss,
#                     "val_loss": self.val_loss,
#                     "train_acc": self.train_acc,
#                     "val_acc": self.val_acc,
#                 })
    
#     def train_epoch(self) -> None:
#         train_loss = []
#         train_acc = []
#         for i, (images, labels) in enumerate(self.train_loader, start=1):
#             # self.plot_samples(images, labels)
#             preds = self.model(images)
#             self.loss_fn(preds, labels)
#             self.optimizer.zero_grad()
#             self.loss_fn.backward()
#             self.optimizer.step(i)
#             train_loss.append(self.loss_fn.value)
#             train_acc.append(np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
#         self.train_loss = np.mean(train_loss)
#         self.train_acc = np.mean(train_acc)
    
#     def val_epoch(self) -> None:
#         val_loss = []
#         val_acc = []
#         for i, (images, labels) in enumerate(self.val_loader):
#             preds = self.model(images)
#             self.loss_fn(preds, labels)
#             val_loss.append(self.loss_fn.value)
#             val_acc.append(np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
#         self.val_loss = np.mean(val_loss)
#         self.val_acc = np.mean(val_acc)
    
#     def plot_images(self, data, labels, class_names=None, flatten=False, use_wandb=False):

#         uniq_labels = np.unique(labels)

#         fig, ax = plt.subplots(2,5, figsize=(15, 6))
#         fig.subplots_adjust(wspace=0.3, hspace=0.3)
#         ax = ax.reshape(-1)

#         for i, label in enumerate(uniq_labels):
#             img = data[np.where(labels == label)[0][0]]
#             if class_names:
#                 ax[i].set_title(class_names[label])
#             if flatten:
#                 img = img.reshape(28, 28)
#             ax[i].imshow(img, cmap='gray')
#             ax[i].axis('off')
#         if use_wandb:
#             wandb.log({"Class Images": fig})
#         plt.close(fig)

#     def sweep(self) -> None:
#         config = wandb.config
#         self.hyperparameters = {**self.hyperparameters, **config}
#         self.set_model(config.hidden_sizes, config.hidden_activations)
#         self.set_optimizer(config.optimizer)
#         self.set_loss_fn(config.loss_fn)
#         wandb.run_name = f"{len(config.hidden_sizes)}Layer-{config.hidden_activations}Activated-{config.optimizer}Optimized-{config.loss_fn}Loss"
#         self.train()

#     def run(self) -> None:
#         if self.do_sweep:
#             self.perform_sweep()
#         else:
#             self.train()

#     def predict(self, x) -> np.ndarray:
#         return np.argmax(self.model(x), axis=1)
    
#     def perform_sweep(self) -> None:
#         with open(self.conf_path, 'r') as f:
#             sweep_config = json.load(f)
#         sweep_id = wandb.sweep(sweep_config, project=self.wandb_project_id, entity=self.wandb_entity)
#         wandb.agent(sweep_id, function=self.sweep)
main()