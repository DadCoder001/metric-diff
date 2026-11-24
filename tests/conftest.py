from metric_diff.backends import _ensure_keras_backend

# Tests import keras directly; pick the torch backend first when TensorFlow isn't installed,
# exactly as metric-diff does before its own first keras import.
_ensure_keras_backend()
