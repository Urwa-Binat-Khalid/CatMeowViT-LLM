# model.py

import timm
import torch.nn as nn

class ViTBase(nn.Module):
    """
    A Vision Transformer (ViT) model using timm's pretrained ViT-Base.
    The head is replaced to match the number of output classes.
    """

    def __init__(self, num_classes):
        """
        Initialize the ViTBase model.

        Args:
            num_classes (int): Number of output classes for classification.
        """
        super().__init__()

        # Load ViT-Base with patch size 16 and input size 224x224.
        # Use pretrained ImageNet weights and set input channels to 3 (RGB).
        self.m = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            in_chans=3
        )

        # Replace the classification head to match the target classes
        self.m.head = nn.Linear(self.m.head.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input image tensor of shape [batch_size, 3, 224, 224].

        Returns:
            Tensor: Predicted class logits.
        """
        return self.m(x)
