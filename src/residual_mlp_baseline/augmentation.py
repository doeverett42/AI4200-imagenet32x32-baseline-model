import torch
import torch.nn.functional as F

#Slightly change each image's color balance with one gain per RGB channel
def random_rgb_jitter(images, strength):
    if not 0 <= strength <= 1:
        raise ValueError("rgb_jitter must be between 0 and 1")
    if strength == 0:
        return images
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("rgb_jitter requires images shaped (batch, 3, height, width)")
    if not images.is_floating_point():
        raise ValueError("rgb_jitter expects prepared floating-point images")

    #Return to [0, 1] so multiplying changes channel intensity around black
    rgb = (images + 1) / 2
    #Each image gets separate R, G, and B gains, shared across all its pixels
    gains = images.new_empty((images.shape[0], 3, 1, 1)).uniform_(1 - strength, 1 + strength)
    #Keep valid pixel values, then restore prepare()'s [-1, 1] range
    return (rgb * gains).clamp(0, 1) * 2 - 1


#Rotate and shift each image independently
def random_affine(images, rotation=0, translate=0, padding_mode="zeros"):
    """
    Expected input:
        Floating-point images already processed by prepare(), so their values are approximately between -1 and 1

        Shape: (batch_size, channels, height, width)

        Examples:
            MNIST:      (256, 1, 28, 28)
            CIFAR-10:   (256, 3, 32, 32)
            ImageNet32: (256, 3, 32, 32)

    Arguments:
        rotation:
            Maximum rotation in degrees
            A value of 7 selects an angle between -7 and +7 degrees independently for each image

        translate:
            Maximum movement in whole pixels along each axis
            A value of 2 allows shifts of -2, -1, 0, 1, or 2 pixels

        padding_mode:
            Determines what happens when the transformation exposes areas outside the original image

            "zeros":     Fill with black after  normalization adjustment
            "reflection": Reflect nearby image content across the boundary
            "border":    Repeat the nearest border pixel

    Returns:
        Transformed images with the same shape as the input.

    Each image gets its own transformation, but all channels belonging to that image share the transformation so RGB colors stay aligned.
    """
    batch_size, _, height, width = images.shape

    #Select an angle in degrees for each image
    angles = images.new_zeros(batch_size)
    if rotation > 0:
        angles.uniform_(-rotation, rotation)
    #PyTorch's sine and cosine functions expect radians, not degrees
    angles = angles * (torch.pi / 180)

    #Select horizontal and vertical shifts in pixels
    shifts = images.new_zeros((batch_size, 2))
    if translate > 0:
        shifts = torch.randint(
            -translate,
            translate + 1,
            (batch_size, 2),
            device=images.device
        )

    cosine, sine = angles.cos(), angles.sin()

    #Each image needs a 2-by-3 transformation matrix:
    #
    #     [a  b  horizontal_offset]
    #     [c  d  vertical_offset  ]
    #
    #The first two columns describe rotation
    #The last column incorporates the selected movement
    transform = images.new_zeros((batch_size, 2, 3))

    #The sampler works backward:
    #For each OUTPUT pixel, it asks where to read the ORIGINAL image
    #Therefore, this describes the inverse of the desired image motion
    transform[:, 0, 0] = cosine
    transform[:, 0, 1] = sine * height / width
    transform[:, 1, 0] = -sine * width / height
    transform[:, 1, 1] = cosine

    #Incorporate the shifts into that backward sampling transformation
    #The sampler uses normalized coordinates spanning a width of 2,
    #rather than ordinary pixel coordinates. With align_corners=False:
    #     one horizontal pixel corresponds to 2 / width
    #     one vertical pixel corresponds to 2 / height
    #The sine/cosine terms combine movement with the inverse rotation
    transform[:, 0, 2] = (-2 * (cosine * shifts[:, 0] + sine * shifts[:, 1]) / width)
    transform[:, 1, 2] = (2 * (sine * shifts[:, 0] - cosine * shifts[:, 1]) / height)

    #Convert the matrices into a sampling map
    #For every output pixel, this map specifies the location to read
    #from the original image. images.size() preserves the output shape
    grid = F.affine_grid(transform, images.size(), align_corners=False)

    #Read the original images using the sampling map
    #
    #"bilinear" blends neighboring pixels when a sampling location
    #falls between them, which commonly happens during rotation
    #
    #Black is -1 after prepare()
    return F.grid_sample(
        images + 1,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False
    ) - 1

#Augment prepared training images shaped (batch, channels, H, W)
def augment_batch(images, config=None):
    """
    Apply this task's optional training augmentations

    Expected images:
        Prepared floating-point tensors in approximately [-1, 1], shaped (batch_size, channels, height, width)

    Example RGB configuration:
        {
            "rotation": 7,
            "translate": 2,
            "hflip_prob": 0.0,
            "rgb_jitter": 0.05,
            "noise_std": 0.02,
            "padding_mode": "zeros"
        }

    Operations run in this order:
        1. Horizontal flipping
        2. Rotation and movement
        3. RGB channel scaling, if enabled on three-channel images
        4. Gaussian pixel noise

    This function changes images, not labels, and does not modify the original input tensor in place
    """
    # An empty configuration leaves the images unchanged.
    if not config:
        return images
    
    #Missing settings default to disabled operations
    rotation = config.get("rotation", 0)
    translate = config.get("translate", 0)
    flip_prob = config.get("hflip_prob", 0.0)
    noise_std = config.get("noise_std", 0.0)
    rgb_jitter = config.get("rgb_jitter", 0.0)
    padding_mode = config.get("padding_mode", "zeros")

    if rotation < 0 or translate < 0 or noise_std < 0:
        raise ValueError("rotation, translate, and noise_std must be nonnegative")

    if not 0 <= flip_prob <= 1:
        raise ValueError("hflip_prob must be between 0 and 1")
    if not 0 <= rgb_jitter <= 1:
        raise ValueError("rgb_jitter must be between 0 and 1")

    # ---------------------------------------------------------------
    # 1. Horizontal flipping
    # ---------------------------------------------------------------
    #
    #A probability of 0.5 gives each image a 50% chance of flipping
    if flip_prob > 0:
        flip_mask = torch.rand(images.shape[0], 1, 1, 1, device=images.device) < flip_prob
        images = torch.where(flip_mask, images.flip(-1), images)

    # ---------------------------------------------------------------
    # 2. Rotation and movement
    # ---------------------------------------------------------------
    #
    if rotation > 0 or translate > 0:
        images = random_affine(images, rotation, translate, padding_mode)

    # ---------------------------------------------------------------
    # 3. RGB channel scaling
    # ---------------------------------------------------------------
    #A strength of 0.05 gives each channel a gain between 0.95 and 1.05
    if rgb_jitter > 0:
        images = random_rgb_jitter(images, rgb_jitter)

    # ---------------------------------------------------------------
    # 4. Gaussian noise
    # ---------------------------------------------------------------
    #
    #Independent pixel noise may discourage dependence on exact pixel values
    if noise_std > 0:
        images = (images + torch.randn_like(images) * noise_std).clamp(-1, 1)

    return images