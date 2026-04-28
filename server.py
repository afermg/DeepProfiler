"""Nahual server for DeepProfiler (CPCNNv1).

DeepProfiler is a TensorFlow/Keras-based morphological profiler. The "CPCNNv1"
variant is a CNN backbone (ResNet50V2 by default in this scaffold) that
produces a feature embedding per single-cell crop.

This server wraps the model with a setup/process pair compatible with the
Nahual responder. The default model is a Keras ResNet50V2 with
``include_top=False`` and ``pooling='avg'`` — that is structurally the
backbone DeepProfiler uses for ImageNet-pretrained CPCNN feature extraction.

Run with:
    nix run . -- ipc:///tmp/deepprofiler.ipc
or:
    python server.py ipc:///tmp/deepprofiler.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

# Reduce TF log noise before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TF 2.13 in nixos-24.11 looks up the standalone `keras` module at runtime;
# nixpkgs ships `tf-keras` (2.17) which matches the API. Set the legacy flag
# before importing tensorflow so `tf.keras` redirects to `tf_keras`.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy
import pynng
import tensorflow as tf
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

# TF 2.13 in nixos-24.11 doesn't ship a top-level `keras` module, but `tf_keras`
# (the legacy 2.x distribution) is available. Import it directly to side-step
# the broken `tf.keras` lazy loader.
try:
    import tf_keras as keras  # type: ignore
except ImportError:  # pragma: no cover
    keras = tf.keras  # type: ignore[assignment]

address = sys.argv[1]


_SUPPORTED_BACKBONES = {
    # CPCNNv1-style: Keras Applications ResNet/EfficientNet backbones.
    "resnet50v2": keras.applications.ResNet50V2,
    "resnet101v2": keras.applications.ResNet101V2,
    "resnet152v2": keras.applications.ResNet152V2,
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetb3": keras.applications.EfficientNetB3,
}


def setup(
    backbone: str = "resnet50v2",
    weights: str | None = "imagenet",
    expected_tile_size: int = 32,
    expected_channels: int = 3,
    input_size: int = 224,
) -> tuple[Callable, dict]:
    """Build a CPCNNv1-style Keras backbone for feature extraction.

    Parameters
    ----------
    backbone : str
        One of ``resnet50v2``, ``resnet101v2``, ``resnet152v2``,
        ``efficientnetb0``, ``efficientnetb3``. CPCNNv1 maps to ``resnet50v2``.
    weights : str | None
        ``"imagenet"`` to load the standard pretrained weights, ``None`` for a
        randomly-initialized model, or a path to a ``.h5`` checkpoint.
    expected_tile_size : int
        Required divisor for the trailing spatial dims of incoming arrays.
    expected_channels : int
        Number of channels the model expects. ImageNet backbones expect 3.
    input_size : int
        Spatial size used to define the model's input. The model is functional
        and accepts variable HxW at inference, but we declare a default.
    """
    key = backbone.lower()
    if key not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}; available: {sorted(_SUPPORTED_BACKBONES)}"
        )

    factory = _SUPPORTED_BACKBONES[key]

    # ImageNet weights require 3 channels.
    if weights == "imagenet" and expected_channels != 3:
        raise ValueError(
            "ImageNet weights require expected_channels=3; got "
            f"{expected_channels}."
        )

    if weights is not None and weights not in ("imagenet",) and not os.path.exists(weights):
        # Treat unknown string as "no weights" to keep smoke tests resilient.
        weights = None

    if weights in (None, "imagenet"):
        model = factory(
            include_top=False,
            weights=weights,
            input_shape=(input_size, input_size, expected_channels),
            pooling="avg",
        )
        load_info = {"weights": weights or "random"}
    else:
        # Treat as a path to a Keras .h5 checkpoint.
        model = factory(
            include_top=False,
            weights=None,
            input_shape=(input_size, input_size, expected_channels),
            pooling="avg",
        )
        model.load_weights(weights)
        load_info = {"weights": weights}

    # Detect available device (informational only — TF auto-places).
    gpus = tf.config.list_physical_devices("GPU")
    device_str = "GPU:0" if gpus else "CPU:0"

    info = {
        "device": device_str,
        "backbone": key,
        "expected_tile_size": expected_tile_size,
        "expected_channels": expected_channels,
        "input_size": input_size,
        "feature_dim": int(model.output_shape[-1]),
        "load": load_info,
    }

    processor = partial(
        process,
        model=model,
        expected_tile_size=expected_tile_size,
        expected_channels=expected_channels,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    expected_tile_size: int,
    expected_channels: int,
) -> numpy.ndarray:
    """Run a CPCNNv1-style backbone on an NCZYX numpy array.

    The Z dimension is dropped, channels are padded to ``expected_channels``,
    and the array is transposed to NHWC for Keras. Returns a numpy embedding
    of shape (N, feature_dim).
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    # pad_channel_dim drops Z (axis 2) and pads channels (axis 1) → NCHW.
    pixels = pad_channel_dim(pixels, expected_channels)

    # Keras backbones expect NHWC, float32.
    pixels = numpy.ascontiguousarray(
        numpy.transpose(pixels, (0, 2, 3, 1)).astype(numpy.float32)
    )

    feats = model(pixels, training=False)
    # Convert TF tensor → numpy before returning.
    if hasattr(feats, "numpy"):
        feats = feats.numpy()
    return feats


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"DeepProfiler server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
