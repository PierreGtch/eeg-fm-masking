import abc
import inspect
from typing import TypeVar, Any, Generic, Sequence

from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ImportString,
    model_validator,
)


class Dummy:
    def __init__(self, name: str = "dummy"):
        self.name = name


class DummyPositional:
    def __init__(self, name: str):
        self.name = name


class DummyKwargs:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs


T = TypeVar("T")


class InstantiatorConfig(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    @abc.abstractmethod
    def create_instance(self) -> T:
        pass


class PathInstantiatorConfig(InstantiatorConfig):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        validate_assignment=True,
        validate_return=True,
    )

    class_path: ImportString = Field(Dummy)
    init_args: dict[str, Any] = {}
    dict_kwargs: dict[str, Any] | None = None
    check_types: bool = True

    @model_validator(mode="after")
    def validate_args(self):
        sig = inspect.signature(self.class_path)
        for k, v in self.init_args.items():
            if k not in sig.parameters:
                raise ValueError(f"Invalid argument {k} for class {self.class_path}")
            if (
                self.check_types
                and (expected_type := sig.parameters[k].annotation)
                is not inspect.Parameter.empty
            ):
                self.init_args[k] = TypeAdapter(
                    expected_type, module=k
                ).validate_python(v)
        for p in sig.parameters.values():
            if (
                (p.default is p.empty)
                and (p.kind != p.VAR_KEYWORD)
                and (p.name not in self.init_args)
            ):
                raise ValueError(
                    f"Missing required argument {p.name} for class {self.class_path}"
                )
            if p.kind == p.POSITIONAL_ONLY:
                raise ValueError(
                    f"Positional only argument {p.name} for class {self.class_path}"
                )
        if self.dict_kwargs is not None:
            if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                raise ValueError(
                    f"Class {self.class_path} does not accept **kwargs arguments"
                )
        return self

    def create_instance(self):
        return self.class_path(**self.init_args, **(self.dict_kwargs or {}))


def instantiate_optional_list(
    config: None | InstantiatorConfig[T] | Sequence[InstantiatorConfig[T]],
) -> None | T | list[T]:
    if config is None:
        return None
    if isinstance(config, InstantiatorConfig):
        return config.create_instance()
    return [c.create_instance() for c in config]
