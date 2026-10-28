from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.gan import Generator


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "mnist_generator_classroom_epoch_010.pth"
)


@lru_cache(maxsize=1)
def load_generator():
    model = Generator(z_dim=100)

    weights = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(weights, strict=True)
    model.eval()

    return model


def generate_digits(num_samples=16):
    if not 1 <= num_samples <= 64:
        raise ValueError("num_samples must be between 1 and 64.")

    model = load_generator()
    noise = torch.randn(num_samples, 100)

    with torch.no_grad():
        images = model(noise)

    images = ((images + 1) / 2).clamp(0, 1)

    buffer = BytesIO()

    save_image(
        images,
        buffer,
        nrow=min(8, num_samples),
        format="PNG",
    )

    return buffer.getvalue()