import os
import shutil
import tensorflow as tf
import keras

H5_PATH = r"models\tomato_disease_mobilenetv2_abc.h5"
OUT_DIR = r"models\tomato_disease_savedmodel"

print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("Loading:", H5_PATH)

# ✅ Patch only InputLayer: batch_shape -> batch_input_shape
class PatchedInputLayer(keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

# ✅ Load H5 using Keras 3 (safe_mode=False avoids strict deserialization issues)
model = keras.saving.load_model(
    H5_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={"InputLayer": PatchedInputLayer},
)

# ✅ Clean output folder
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# ✅ Export to TF SavedModel in the MOST compatible way
tf.saved_model.save(model, OUT_DIR)

print("✅ SavedModel exported to:", OUT_DIR)
print("✅ Done.")