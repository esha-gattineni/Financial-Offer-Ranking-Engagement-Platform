from pathlib import Path
import shutil

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tensorflow_engagement.keras"
)

EXPORT_PATH = (
    PROJECT_ROOT
    / "serving"
    / "user_engagement"
    / "1"
)


def main():
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nModel inputs:")
    for input_tensor in model.inputs:
        print(
            f"  {input_tensor.name} "
            f"{input_tensor.shape} "
            f"{input_tensor.dtype}"
        )

    if EXPORT_PATH.exists():
        shutil.rmtree(EXPORT_PATH)

    EXPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nExporting SavedModel to:\n{EXPORT_PATH}")

    model.export(EXPORT_PATH)

    print("\nSavedModel export complete.")


if __name__ == "__main__":
    main()