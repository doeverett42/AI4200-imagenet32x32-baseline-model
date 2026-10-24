import torch
import torch.nn as nn


# Each task differs only in its input size and its number of classes.
# The same model class is used for all three.
TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
}

#Give each task independent settings; begin at baseline
TASK_SETTINGS = {
    "mnist": {
        "batch_size": 256,
        "epochs": 15,
        "width": 784,
        "hidden": 1568,
        "depth": 70,
        "activation": "tanh",
        "optimizer": "adamw",
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "step_decay": True
    },

    "cifar10": {
        "batch_size": 256,
        "epochs": 30,
        "width": 128,
        "hidden": 256,
        "depth": 30,
        "activation": "tanh",
        "optimizer": "sgd",
        "lr": 0.05,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "step_decay": False
    },

    "imagenet32": {
        "batch_size": 256,
        "epochs": 10,
        "width": 128,
        "hidden": 256,
        "depth": 30,
        "activation": "tanh",
        "optimizer": "sgd",
        "lr": 0.05,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "step_decay": False
    }
}
#optional activation function changer 
def make_activation(name):
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()


class Block(nn.Module):
    """One residual block:  x + Linear(tanh(Linear(LayerNorm(x))))

    The block expands to `hidden` features, applies tanh, and projects back to
    `width` features so its output can be added to its input (the skip connection).
    """

    def __init__(self, width, hidden, activation="tanh"):
        super().__init__()
        self.activation = make_activation(activation)
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width

    def forward(self, x):
        h = self.fc2(self.activation(self.fc1(self.norm(x))))
        return x + h                          # skip connection: add the input back


class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> tanh -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=128, hidden=256, depth=30, activation="tanh"):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([Block(width, hidden, activation) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_activation = make_activation(activation)
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        #Automatically match initialization to the activation function 
        if activation == "relu": 
            for block in self.blocks: 
                nn.init.kaiming_normal_(block.fc1.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(self.head_fc1.weight, nonlinearity="relu")

    def forward(self, x):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_fc2(self.head_activation(self.head_fc1(self.head_norm(x))))
        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)
