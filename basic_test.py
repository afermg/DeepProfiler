"""Standalone smoke test for the DeepProfiler (CPCNNv1) backbone.

Loads ResNet50V2 with ImageNet weights via tf_keras (the legacy 2.x Keras
distribution that matches TF 2.13's compat surface) and runs a single forward
pass on a random tile. Should print ``(1, 2048)``.

Run with:
    nix develop --impure --command python basic_test.py
"""

import os

# TF 2.13 in nixos-24.11 looks up the standalone `keras` module at runtime;
# nixpkgs ships `tf-keras` (2.17) which matches the API. Set the legacy flag
# before importing tensorflow/keras so `tf.keras` redirects to `tf_keras`.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy
import tf_keras as keras

m = keras.applications.ResNet50V2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3),
)
x = numpy.random.random((1, 224, 224, 3)).astype("float32")
y = m.predict(x)
print(y.shape)
