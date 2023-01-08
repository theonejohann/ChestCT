import tensorflow as tf
import tensorflowjs as tfjs

# Load the TFLite model using the TensorFlow Lite Interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

tfjs.converters.convert_tf_saved_model("/model.tflite", "/")