import torch
import torch.nn as nn
import torch.nn.functional as F


# Each task supplies its input size and number of classes.
# train.py selects the architecture using the task settings.
TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
}

#Give each task independent settings; begin at baseline
TASK_SETTINGS = {
    "mnist": {
        "batch_size": 256,
        "epochs": 40,
        "width": 392,
        "hidden": 1024,
        "depth": 10,
        "activation": "gelu",
        "optimizer": "adamw",
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "scheduler": "cosine",
        "normalization": "batch",
        "head_type": "linear", 
        "block_type": "maxout",
        "maxout_pieces": 2,
        "augmentation": {
            "rotation": 7, 
            "translate": 2, 
            "hflip_prob": 0.0, 
            "noise_std": 0.0, 
            "padding_mode": "zeros"   # Fill exposed areas with black
        }   
    },

    "cifar10": {
        "batch_size": 256,
        "epochs": 90,
        "width": 512,
        "hidden": 1024,
        "depth": 30,
        "activation": "gelu",
        "optimizer": "sgd",
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "dropout": 0.1, 
        "scheduler": "cosine",
        "normalization": "layer",
        "head_type": "mlp",
        "block_type": "mlp",
        "maxout_pieces": 2,
        # Patch dimensions; the top-level dimensions belong to the flat model
        "patch_model": {
            "image_size": 32,
            "patch_size": 4,
            "width": 128,
            "depth": 6,
            "token_hidden": 64,
            "channel_hidden": 256
        },
        "augmentation": {
            "hflip_prob": 0.5,
            "translate": 2, 
            "rotation": 0,
            "noise_std": 0.0,
            "rgb_jitter": 0.05,  #Small independent changes to R, G, and B intensity
            "padding_mode": "reflection"  # Fill exposed edges with reflected pixels    
        }
    },

    "imagenet32": {
        "batch_size": 256,
        "epochs": 20,
        "width": 128,
        "hidden": 256,
        "depth": 15,
        "activation": "gelu",
        "optimizer": "sgd",
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "dropout": 0,
        "scheduler": "none",
        # Patch dimensions; the top-level dimensions belong to the flat model
        "patch_model": {
            "image_size": 32,
            "patch_size": 4,
            "width": 128,
            "depth": 12,
            "token_hidden": 64,
            "channel_hidden": 256
        },
        "augmentation": {
            "hflip_prob": 0.5, 
            "translate": 0,
            "noise_std": 0.02,
            "padding_mode": "reflection"
        }
    }
}
#optional activation function changer 
def make_activation(name):
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()

#choose normalization for a newly created, untrained model
def configure_normalization(model, normalization="layer"):
    #The original model already contains LayerNorm
    if normalization == "layer":
        return model

    if normalization != "batch":
        raise ValueError(f"Unknown normalization: {normalization}")

    width = model.input_layer.out_features

    #Replace normalization inside each residual branch
    for block in model.blocks:
        block.norm = nn.BatchNorm1d(width)

    #Replace the normalization before the classification head
    model.head_norm = nn.BatchNorm1d(width)

    return model

#Choose the head for a newly created, untrained model
def configure_head(model, head_type="mlp"):
    #Keep the original two-layer classification head
    if head_type == "mlp":
        return model

    if head_type != "linear":
        raise ValueError(f"Unknown head type: {head_type}")

    width = model.input_layer.out_features
    num_classes = model.head_fc2.out_features

    #Bypass the extra hidden layer and its activation
    model.head_fc1 = nn.Identity()
    model.head_activation = nn.Identity()

    #Classify the backbone features directly
    model.head_fc2 = nn.Linear(width, num_classes)

    #This layer was created after the constructor's initialization loop
    nn.init.xavier_normal_(model.head_fc2.weight)
    nn.init.zeros_(model.head_fc2.bias)

    return model

#Keep the largest candidate for each hidden feature
#Idea from paper: https://proceedings.mlr.press/v28/goodfellow13.pdf
class Maxout(nn.Module):
    def __init__(self, pieces=2):
        super().__init__()

        if not isinstance(pieces, int) or pieces < 2:
            raise ValueError("pieces must be an integer of at least 2")

        self.pieces = pieces

    def forward(self, x):
        #Preserve batch and patch dimensions while grouping candidates
        x = x.reshape(*x.shape[:-1], -1, self.pieces)

        #Keep the largest candidate in each group
        return x.max(dim=-1).values

#Multiply learned values by learned SiLU gates
class SwiGLU(nn.Module):
    def forward(self, x):
        #fc1 produces two halves, each containing hidden features
        gate, value = x.chunk(2, dim=-1)

        #Apply SiLU to the gate, then multiply corresponding features
        return F.silu(gate) * value

#Choose the hidden computation for a fresh, untrained model
def configure_blocks(model, block_type="mlp", maxout_pieces=2):
    if block_type == "mlp":
        return model

    if block_type not in ("maxout", "swiglu"):
        raise ValueError(f"Unknown block type: {block_type}")

    for block in model.blocks:
        width = block.fc1.in_features
        hidden = block.fc2.in_features

        if block_type == "maxout":
            #Generate candidates; Maxout keeps the largest per feature
            block.activation = Maxout(maxout_pieces)
            projected_features = hidden * maxout_pieces
        else:
            #Generate one gate and one value per hidden feature
            block.activation = SwiGLU()
            projected_features = 2 * hidden

        block.fc1 = nn.Linear(width, projected_features)

        #This replacement was created after the constructor's init loop
        nn.init.xavier_normal_(block.fc1.weight)
        nn.init.zeros_(block.fc1.bias)

    return model


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
        # Inference pass, never used during training. Returns class probabilities.
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)
