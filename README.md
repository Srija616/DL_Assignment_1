# DL_Assignment_1
To run the training, please run the following command from the assignment_1 directory. This will use the default configuration
```python
python train.py --wandb_entity myname --wandb_project myprojectname
```

Arguments to be given:
```python
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
```
