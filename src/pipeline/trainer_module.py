import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio


LABEL_KEY = "clicked"


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=4096):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    transformed_feature_spec.pop(LABEL_KEY)

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=LABEL_KEY,
        ),
        schema=tf_transform_output.transformed_metadata.schema,
    )

    return dataset


def _build_model(tf_transform_output):
    feature_spec = tf_transform_output.transformed_feature_spec()

    inputs = {}
    encoded_features = []

    for feature_name, spec in feature_spec.items():
        if feature_name == LABEL_KEY:
            continue

        if spec.dtype == tf.float32:
            inp = tf.keras.Input(
                shape=(1,),
                name=feature_name,
                dtype=tf.float32,
            )
            inputs[feature_name] = inp
            encoded_features.append(inp)

        elif spec.dtype in (tf.int64, tf.int32):
            inp = tf.keras.Input(
                shape=(1,),
                name=feature_name,
                dtype=spec.dtype,
            )
            inputs[feature_name] = inp

            vocab_size = tf_transform_output.vocabulary_size_by_name(
                feature_name.replace("_xf", "")
            )

            embedding_dim = min(32, max(4, int(vocab_size ** 0.5)))

            embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size + 1,
                output_dim=embedding_dim,
            )(inp)

            embedding = tf.keras.layers.Flatten()(embedding)
            encoded_features.append(embedding)

    x = tf.keras.layers.Concatenate()(encoded_features)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    return model


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
    )

    model = _build_model(tf_transform_output)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
            )
        ],
    )

    transform_features_layer = tf_transform_output.transform_features_layer()

    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(LABEL_KEY)

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        raw_features = tf.io.parse_example(
            serialized_tf_examples,
            raw_feature_spec,
        )

        raw_features[LABEL_KEY] = tf.zeros(
            shape=(tf.shape(serialized_tf_examples)[0],),
            dtype=tf.int64,
        )

        transformed_features = transform_features_layer(raw_features)
        transformed_features.pop(LABEL_KEY)

        return {
            "predictions": model(transformed_features)
        }

    signatures = {
        "serving_default": serve_tf_examples_fn.get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples",
            )
        )
    }

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures=signatures,
    )