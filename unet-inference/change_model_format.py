import tensorflow as tf 

model = tf.keras.models.load_model("C:/Users/socho/Documents/GCP_implementation/models/model_for_nuclei.h5")
model.export("unet-good-format")