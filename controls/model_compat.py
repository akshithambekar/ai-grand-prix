"""Explicit policy/observation compatibility checks."""

from controls.observation import OBSERVATION_SCHEMA, OBSERVATION_SIZE


def stamp_model_schema(model):
    model.observation_schema = OBSERVATION_SCHEMA
    return model


def validate_model_schema(model):
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    schema = getattr(model, "observation_schema", None)
    if shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"incompatible PPO observation: model shape={shape}, required "
            f"{OBSERVATION_SCHEMA} shape=({OBSERVATION_SIZE},); retrain from scratch"
        )
    if schema not in (None, OBSERVATION_SCHEMA):
        raise ValueError(
            f"incompatible PPO observation schema {schema!r}; required {OBSERVATION_SCHEMA!r}"
        )
    return model
