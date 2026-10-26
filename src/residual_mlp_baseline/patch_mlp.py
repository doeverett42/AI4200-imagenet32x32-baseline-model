# Patch Residual MLP
#
# I want to see whether processing smaller image regions helps the model learn useful patterns more efficiently
#
# With 4x4 patches, a 32x32 RGB image becomes 64 patches. A shared Linear
# layer projects each patch into learned features. Each residual block
# mixes information between patches, then mixes features within a patch.
#
# Activation, normalization, branch type, head type, and dropout come from
# the task's settings. Patch size, feature width, and depth are configurable.
# TASKS supplies the class count: 10 for CIFAR or 1000 for ImageNet32.
# Tokens are patches; channels inside the blocks are learned features.
#
# Source/Credit: https://arxiv.org/abs/2105.01601

import torch
import torch.nn as nn

from .model import make_activation, Maxout, SwiGLU


def make_mlp(inputs, hidden, outputs, dropout, activation="gelu", block_type="mlp", maxout_pieces=2):
    #The activation outputs hidden features; gates need a wider first layer
    if block_type == "mlp":
        expansion, nonlinear = 1, make_activation(activation)
    elif block_type == "maxout":
        #Produce several candidates for each feature; Maxout keeps the largest
        expansion, nonlinear = maxout_pieces, Maxout(maxout_pieces)
    elif block_type == "swiglu":
        #Produce one learned gate and one learned value per hidden feature
        expansion, nonlinear = 2, SwiGLU()
    else:
        raise ValueError(f"Unknown block type: {block_type}")

    #Linear transforms the last dimension and preserves the leading dimensions
    return nn.Sequential(
        nn.Linear(inputs, hidden * expansion),
        nonlinear,
        nn.Dropout(dropout),
        nn.Linear(hidden, outputs)
    )

#Normalize the last feature dimension of a patch tensor
class FeatureNorm(nn.Module):
    def __init__(self, width, normalization):
        super().__init__()
        if normalization == "layer":
            self.norm = nn.LayerNorm(width)
        elif normalization == "batch":
            self.norm = nn.BatchNorm1d(width)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

    def forward(self, x):
        shape = x.shape
        #Reshape (batch, patches, features) into (batch * patches, features)
        #LayerNorm works per row; BatchNorm uses all rows for each feature
        return self.norm(x.reshape(-1, shape[-1])).reshape(shape)

#Return one residual update with the same shape as the input
class MixingBranch(nn.Module):
    def __init__(self, patches, width, hidden, mix_tokens, activation="gelu", normalization="layer", block_type="mlp", maxout_pieces=2, dropout=0.0):
        super().__init__()
        self.mix_tokens = mix_tokens
        self.norm = FeatureNorm(width, normalization)
        #The branch must return the same size so its update can be added back
        features = patches if mix_tokens else width
        self.mlp = make_mlp(features, hidden, features, dropout, activation, block_type, maxout_pieces)

    def forward(self, x):
        #Normalize learned features before changing the axis being mixed
        h = self.norm(x)
        if self.mix_tokens:
            # (batch, patches, features) -> (batch, features, patches)
            h = h.transpose(1, 2)
        h = self.mlp(h)
        #Both branches return (batch, patches, features) for the skip connection
        return h.transpose(1, 2) if self.mix_tokens else h


class PatchBlock(nn.Module):
    def __init__(self, patches, width, token_hidden, channel_hidden, **options):
        super().__init__()
        self.token_branch = MixingBranch(patches, width, token_hidden, mix_tokens=True, **options)
        self.channel_branch = MixingBranch(patches, width, channel_hidden, mix_tokens=False, **options)

    def forward(self, x):
        #Add learned corrections instead of replacing the existing features
        x = x + self.token_branch(x)
        #Channel mixing receives the features already updated by token mixing
        return x + self.channel_branch(x)

#Patch geometry comes from settings; class count comes from TASKS
class PatchResidualMLP(nn.Module):
    def __init__(self, in_features, num_classes, image_size=32,
                 patch_size=4, width=128, depth=6,
                 token_hidden=64, channel_hidden=256,
                 activation="gelu", normalization="layer",
                 block_type="mlp", maxout_pieces=2,
                 head_type="linear", dropout=0.0):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("patch_size must be positive and divide image_size")
        if in_features <= 0 or in_features % (image_size * image_size) != 0:
            raise ValueError("in_features must match image_size")

        self.image_size = image_size
        self.patch_size = patch_size
        #Recover the number of input color channels from the task metadata
        self.channels = in_features // (image_size * image_size)
        #A 32x32 image with 4x4 patches gives an 8x8 grid: 64 patches
        self.num_patches = (image_size // patch_size) ** 2

        #The same weights turn each patch's pixel values into learned features
        self.patch_projection = nn.Linear(self.channels * patch_size * patch_size, width)
        options = dict(
            activation=activation, normalization=normalization,
            block_type=block_type, maxout_pieces=maxout_pieces,
            dropout=dropout
        )
        #Each block has its own weights; the blocks are not one reused module
        self.blocks = nn.ModuleList([PatchBlock(self.num_patches, width, token_hidden, channel_hidden, **options) for _ in range(depth)])

        self.head_norm = FeatureNorm(width, normalization)
        #num_classes creates ten CIFAR scores or 1000 ImageNet32 scores
        if head_type == "linear":
            self.head = nn.Linear(width, num_classes)
        elif head_type == "mlp":
            #The MLP head uses the ordinary activation, as in the flat model
            # channel_hidden sets the head width; 0.0 disables head dropout
            self.head = make_mlp(width, channel_hidden, num_classes, 0.0, activation)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        # Initialize every Linear layer before applying the ReLU adjustment below
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        #Apply He only to matrices actually feeding an ordinary ReLU
        if activation == "relu":
            if block_type == "mlp":
                for block in self.blocks:
                    for branch in (block.token_branch, block.channel_branch):
                        nn.init.kaiming_normal_(branch.mlp[0].weight, nonlinearity="relu")
            if head_type == "mlp":
                nn.init.kaiming_normal_(self.head[0].weight, nonlinearity="relu")

    def make_patches(self, x):
        #Prepared images use (batch, color channels, height, width)
        expected = (self.channels, self.image_size, self.image_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected image dimensions {expected} after batch")
        batch, channels, height, width = x.shape
        p = self.patch_size
        #Separate grid rows/columns from the pixels inside each patch
        x = x.reshape(batch, channels, height // p, p, width // p, p)
        #Group patches first, then keep each patch's channels and pixels together
        x = x.permute(0, 2, 4, 1, 3, 5)
        #For 32x32 RGB images with 4x4 patches, return (batch, 64, 48)
        return x.reshape(batch, self.num_patches, channels * p * p)

    def forward(self, x):
        #Project each patch into width features, for example (batch, 64, 128)
        x = self.patch_projection(self.make_patches(x))
        for block in self.blocks:
            x = block(x)
        #Normalize first, then average patches into one feature vector per image
        x = self.head_norm(x).mean(dim=1)
        #CrossEntropyLoss expects raw class scores during training
        return self.head(x)

    def predict(self, x):
        #Disable dropout and use stored BatchNorm statistics for inference
        self.eval()
        #Prediction needs probabilities without building a gradient graph
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)