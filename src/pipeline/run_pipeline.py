from pathlib import Path

import tfx.v1 as tfx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_ROOT = PROJECT_ROOT / "artifacts" / "tfx_pipeline"
DATA_ROOT = PROJECT_ROOT / "data" / "tfx_input"
MODULE_FILE = PROJECT_ROOT / "src" / "pipeline" / "trainer_module.py"
PREPROCESSING_FN = "src.transform.preprocessing.preprocessing_fn"

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tfx_metadata"
    / "metadata.db"
)


def create_pipeline():
    example_gen = tfx.components.CsvExampleGen(
        input_base=str(DATA_ROOT),
        input_config=tfx.proto.Input(
            splits=[
                tfx.proto.Input.Split(
                    name="train",
                    pattern="train/*",
                ),
                tfx.proto.Input.Split(
                    name="eval",
                    pattern="eval/*",
                ),
            ]
        ),
    )

    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"],
    )

    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn=PREPROCESSING_FN,
        splits_config=tfx.proto.SplitsConfig(
            analyze=["train"],
            transform=["train", "eval"],
        ),
    )

    trainer = tfx.components.Trainer(
        module_file=str(MODULE_FILE),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=tfx.proto.TrainArgs(
            splits=["train"],
            num_steps=2000,
        ),
        eval_args=tfx.proto.EvalArgs(
            splits=["eval"],
            num_steps=500,
        ),
    )

    return tfx.dsl.Pipeline(
        pipeline_name="user_engagement_pipeline",
        pipeline_root=str(PIPELINE_ROOT),
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
        ],
        metadata_connection_config=
            tfx.orchestration.metadata.sqlite_metadata_connection_config(
                str(METADATA_PATH)
            ),
        beam_pipeline_args=[
            "--runner=DirectRunner",
            "--update_compatibility_version=2.67.0",
            "--direct_num_workers=1",
            "--direct_running_mode=in_memory",
        ],
    )


if __name__ == "__main__":
    pipeline = create_pipeline()

    tfx.orchestration.LocalDagRunner().run(pipeline)