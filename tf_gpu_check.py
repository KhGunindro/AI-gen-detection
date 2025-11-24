# tf_gpu_check.py
import tensorflow as tf
import platform, sys
print("Platform:", platform.platform())
print("Python:", sys.version)
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

try:
    for g in gpus:
        details = tf.config.experimental.get_device_details(g)
        print("GPU details:", details)
except Exception as e:
    print("Error getting device details:", e)