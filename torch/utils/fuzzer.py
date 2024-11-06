import copy
import itertools
import logging
import random
import string
import sys
import traceback
from functools import wraps
from typing import Any, Callable, get_args, get_origin, Literal, Optional, Union

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass


def is_optional_type(type_hint) -> bool:
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


def is_callable_type(type_hint) -> bool:
    return type_hint.__name__ == "Callable"


class DummyPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """
        return None

    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """
        return None


TYPE_EXEMPLARS: dict[str, Any] = {
    CustomGraphPass.__name__: DummyPass(),
    torch.fx.graph.Graph.__name__: torch.fx.graph.Graph(),
    torch._inductor.scheduler.BaseSchedulerNode.__name__: torch._inductor.scheduler.BaseSchedulerNode(
        None
    ),
}


class SamplingMethod:
    TOGGLE = 1  # toggle to the opposite value
    RANDOM = 2  # randomly choose an option

    def _generate_toggle_value_for_type(type_hint: type, default: Any) -> Any:
        """this toggle setting will use randomness too, but if there's a sensible 'toggle', it will use that"""
        if type_hint == bool:
            return not default
        elif type_hint == int:
            return -default if default != 0 else random.randint(-100, 100)
        elif type_hint == float:
            return random.uniform(-100, 100)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif get_origin(type_hint) is list:
            elem_type = type_hint.__args__[0]
            new_default = default[0] if len(default) > 0 else None
            return [
                SamplingMethod._generate_toggle_value_for_type(elem_type, new_default)
                for _ in range(random.randint(0, 3))
            ]
        elif get_origin(type_hint) is dict:
            key_type, value_type = type_hint.__args__
            items = list(default.items())
            if len(items) > 0:
                default_key, default_val = items[0]
                return {
                    SamplingMethod._generate_toggle_value_for_type(
                        key_type, default_key
                    ): SamplingMethod._generate_toggle_value_for_type(
                        value_type, default_val
                    )
                    for _ in range(random.randint(0, 3))
                }
            else:
                # fall back to random
                return {
                    SamplingMethod._generate_random_value_for_type(
                        key_type, None
                    ): SamplingMethod._generate_random_value_for_type(value_type, None)
                    for _ in range(random.randint(0, 3))
                }
        elif get_origin(type_hint) is Union:
            # do whatever is not the type of default
            assert len(type_hint.__args__) > 1
            new_type = random.choice(
                [t for t in type_hint.__args__ if t != type(default)]
            )
            try:
                return SamplingMethod._generate_random_value_for_type(
                    new_type, new_type()
                )
            except:
                # if default constructor doesn't work, try None
                try:
                    return SamplingMethod._generate_random_value_for_type(
                        new_type, None
                    )
                except:
                    breakpoint()
                    return default
        elif get_origin(type_hint) is tuple:
            zipped = zip(type_hint.__args__, default)
            return tuple(
                map(
                    lambda x: SamplingMethod._generate_toggle_value_for_type(
                        x[0], x[1]
                    ),
                    zipped,
                )
            )
        elif get_origin(type_hint) is Literal:
            return random.choice([t for t in type_hint.__args__ if t != type(default)])
        elif is_optional_type(type_hint):
            elem_type = type_hint.__args__[0]
            if default is None:
                return SamplingMethod._generate_random_value_for_type(elem_type)
            else:
                return None
        elif type_hint is type(None):
            # needed for recursive calls
            return None
        elif is_callable_type(type_hint):
            input_args, return_type = (
                list(type_hint.__args__)[:-1],
                list(type_hint.__args__)[-1],
            )

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs) -> return_type:
                return SamplingMethod._generate_random_value_for_type(return_type)

            return dummy_function
        elif type_hint.__name__ in TYPE_EXEMPLARS:
            return TYPE_EXEMPLARS[type_hint.__name__]
        elif type_hint == Any:
            return 1 if not default == 1 else 2
        else:
            breakpoint()
            raise Exception(f"Unable to process type {type_hint}. PRs welcome :)")

    def _generate_random_value_for_type(type_hint: type, _default: Any = None) -> Any:
        """Generate a random value for a given type."""
        if type_hint == bool:
            return random.choice([True, False])
        elif type_hint == int:
            return random.randint(-100, 100)
        elif type_hint == float:
            return random.uniform(-100, 100)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif get_origin(type_hint) is list:
            elem_type = type_hint.__args__[0]
            return [
                SamplingMethod._generate_random_value_for_type(elem_type)
                for _ in range(random.randint(0, 3))
            ]
        elif get_origin(type_hint) is dict:
            key_type, value_type = type_hint.__args__
            return {
                SamplingMethod._generate_random_value_for_type(
                    key_type
                ): SamplingMethod._generate_random_value_for_type(value_type)
                for _ in range(random.randint(0, 3))
            }
        elif get_origin(type_hint) is Union:
            return SamplingMethod._generate_random_value_for_type(
                random.choice(type_hint.__args__)
            )
        elif get_origin(type_hint) is tuple:
            return tuple(
                map(SamplingMethod._generate_random_value_for_type, type_hint.__args__)
            )
        elif get_origin(type_hint) is Literal:
            return random.choice(type_hint.__args__)
        elif is_optional_type(type_hint):
            elem_type = type_hint.__args__[0]
            return random.choice(
                [None, SamplingMethod._generate_random_value_for_type(elem_type)]
            )
        elif type_hint is type(None):
            return None
        elif is_callable_type(type_hint):
            input_args, return_type = (
                list(type_hint.__args__)[:-1],
                list(type_hint.__args__)[-1],
            )

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs) -> return_type:
                return SamplingMethod._generate_random_value_for_type(return_type)

            return dummy_function
        elif type_hint.__name__ in TYPE_EXEMPLARS:
            return TYPE_EXEMPLARS[type_hint.__name__]
        elif type_hint == Any:
            return 1
        else:
            breakpoint()
            raise Exception(f"Unable to process type {type_hint}. PRs welcome :)")

    def dispatch(sm: "SamplingMethod"):
        if sm == SamplingMethod.RANDOM:
            return SamplingMethod._generate_random_value_for_type
        elif sm == SamplingMethod.TOGGLE:
            return SamplingMethod._generate_toggle_value_for_type
        else:
            raise Exception(f"malformed sampling method: {sm}")


# TODO create a matrix of the results


class ConfigFuzzer:
    def __init__(
        self,
        config_module,
        test_model_fn_factory: Callable,
        seed: int,
        default: Optional[dict[str, str]] = None,
        sm: SamplingMethod = SamplingMethod.TOGGLE,
    ):
        """
        Initialize the config fuzzer.

        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful
        """
        self.seed = seed
        self.config_module = config_module
        self.test_model_fn_factory = test_model_fn_factory
        self.fields = self.config_module._config
        self.logger = self._setup_logger()
        self.sample = SamplingMethod.dispatch(sm)
        if default is None:
            # these defaults are for inductor, TODO generalize
            self.default = {
                "force_disable_caches": True,
                "cpp.cxx": "clang++" if sys.platform == "darwin" else "g++",
                "TYPE_CHECKING": False,
            }
        else:
            self.default = default

    def __repr__(self):
        return f"ConfigFuzzer(config_module={self.config_module}, test_model_fn={self.test_model_fn}, seed={self.seed}, default={self.default})"

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ConfigFuzzer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _get_type_hint(self, obj, name) -> type:
        """Get type hint for a field, falling back to type(default_value) if not found."""
        try:
            hints = get_type_hints(obj)
            return hints.get(name, type(getattr(obj, name)))
        except Exception:
            return type(getattr(obj, name))

    def _set_config(self, field_name: str, value: Any):
        """Set a config value in the module."""
        setattr(self.config_module, field_name, value)

    def _reset_configs(self):
        """Reset all configs to their default values."""
        for field_name, field_obj in self.fields.items():
            self._set_config(field_name, field_obj.default)

    def fuzz_n_tuple(
        self,
        n: int,
        max_combinations: int = 1000,
    ):
        """Test every combination of n configs."""
        self._reset_configs()
        self.logger.info(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            print(combo)
            config = copy.deepcopy(self.default)
            skip = False
            torch._dynamo.reset()
            for field_name in combo:
                if field_name in config:
                    skip = True
                if field_name.startswith("_"):
                    skip = True
                field = self.fields[field_name]
                value = self.sample(field.value_type, field.default)
                config[field_name] = value
            if skip:
                continue

            test_model_fn = self.test_model_fn_factory()
            comp = torch.compile(options=config)(test_model_fn)

            try:
                success = comp()
                if not success:
                    self.logger.error("Failure with config combination:")
                    for field, value in config.items():
                        self.logger.error(f"{field} = {value}")
                    return False
            except Exception as e:
                traceback.print_exc()
                self.logger.error("Exception with config combination:")
                for field, value in config.items():
                    self.logger.error(f"{field} = {value}")
                self.logger.error(f"Exception: {str(e)}")
                return False

            max_combinations -= 1
            if max_combinations <= 0:
                self.logger.info("Reached maximum combinations limit")
                break

        return True

    def fuzz_random_with_bisect(self, num_attempts: int = 100):
        """Randomly test configs and bisect to minimal failing configuration."""
        # TODO
        self.logger.info(f"Starting random testing with bisection and seed {self.seed}")
        random.seed(self.seed)

        for attempt in range(num_attempts):
            self.logger.info(f"Random attempt {attempt + 1}/{num_attempts}")

            # Generate random configs
            test_configs = []
            for field in self.fields:
                if random.random() < 0.3:  # 30% chance to include each config
                    value = self._generate_random_value_for_type(field.value_type)
                    test_configs.append((field, value))

            # Test the configuration
            self._reset_configs()
            for field, value in test_configs:
                self._set_config(field, value)

            try:
                success = self.test_model_fn()
                if not success:
                    self.logger.info("Found failing configuration, starting bisection")
                    minimal_failing_config = self._bisect_failing_config(test_configs)
                    self.logger.error("Minimal failing configuration:")
                    for field, value in minimal_failing_config:
                        self.logger.error(f"{field.name} = {value}")
                    return False
            except Exception as e:
                self.logger.error(f"Exception during testing: {str(e)}")
                minimal_failing_config = self._bisect_failing_config(test_configs)
                self.logger.error("Minimal failing configuration:")
                for field, value in minimal_failing_config:
                    self.logger.error(f"{field.name} = {value}")
                return False

        self.logger.info("All random tests passed")
        return True

    def _bisect_failing_config(self, failing_configs):
        """Bisect a failing configuration to find minimal set of configs that cause failure."""
        if len(failing_configs) <= 1:
            return failing_configs

        mid = len(failing_configs) // 2
        first_half = failing_configs[:mid]
        second_half = failing_configs[mid:]

        # Test first half
        self._reset_configs()
        for field, value in first_half:
            self._set_config(field, value)

        try:
            if not self.test_model_fn():
                return self._bisect_failing_config(first_half)
        except Exception:
            return self._bisect_failing_config(first_half)

        # Test second half
        self._reset_configs()
        for field, value in second_half:
            self._set_config(field, value)

        try:
            if not self.test_model_fn():
                return self._bisect_failing_config(second_half)
        except Exception:
            return self._bisect_failing_config(second_half)

        # If neither half fails on its own, we need both
        return failing_configs


def create_simple_test_model_cpu():
    """Create a simple test model function for demonstration."""

    def test_fn():
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
            )

            x = torch.randn(32, 10)
            y = model(x)
            return True
        except Exception as e:
            print(f"Model test failed: {str(e)}")
            return False

    return test_fn


def create_simple_test_model_gpu():
    """Create a simple test model function for demonstration."""

    batch_size = 32
    seq_length = 50
    hidden_size = 768

    def test_fn():
        inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
        weight = torch.randn(hidden_size, hidden_size, device="cuda")
        matmul_output = inp @ weight
        final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
        return True

    return test_fn


def main():
    # Example usage
    import torch._inductor.config as cfg

    fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=0)

    # Test every pair of configs
    fuzzer.fuzz_n_tuple(2, max_combinations=100000)

    # Test random configs with bisection
    # fuzzer.fuzz_random_with_bisect(num_attempts=50)


if __name__ == "__main__":
    main()
