import unittest
from types import SimpleNamespace

from controls.model_compat import stamp_model_schema, validate_model_schema


class ModelCompatibilityTests(unittest.TestCase):
    def test_legacy_observation_shape_is_rejected_clearly(self):
        model = SimpleNamespace(observation_space=SimpleNamespace(shape=(19,)))
        with self.assertRaisesRegex(ValueError, "state_v2.*retrain"):
            validate_model_schema(model)

    def test_state_v2_model_is_accepted_and_stamped(self):
        model = SimpleNamespace(observation_space=SimpleNamespace(shape=(38,)))
        stamp_model_schema(model)
        self.assertIs(validate_model_schema(model), model)


if __name__ == "__main__":
    unittest.main()
