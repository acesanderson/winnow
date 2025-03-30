"""
MVP'ing my way to an evaluation framework.
Workflow: a wrapper around an LLM function
Trial: a run of a workflow for a given set of params / configs
Harness: a class to run trials and evaluate them with a single evaluation module
Orchestrator: a class to run multiple trials with multiple evaluation modules
"""

from Mentor import Mentor
from Kramer import get_all_certs
from Chain import Chain, MessageStore
from rich.console import Console
from typing import Callable, Optional
from inspect import signature
from typing import Type
from pydantic import create_model

Chain._message_store = MessageStore()
console = Console()


def inspect_function(func: Callable):
    """
    Inspect a function and return its signature.
    """
    sig = signature(func)
    params = sig.parameters
    return params


class Workflow:
    """
    This defines a workflow: takes an input and returns an output.
    This is like functools.partial, but with a function signature.
    NOTE: the workflow function needs type definitions for its parameters.
    """

    def __init__(self, workflow_function: Callable, params: dict):
        """
        Initialize the workflow with the given function.
        """
        if not callable(workflow_function):
            raise TypeError("Workflow function must be callable.")
        self.workflow_function = workflow_function
        if not self._validate_params(params):
            raise ValueError("Invalid parameters for workflow function.")
        self.name = self.workflow_function.__name__
        self.description = self.workflow_function.__doc__ or ""

    def __call__(self, *args, **kwargs):
        """
        Call the workflow with the given arguments.
        """
        return self.workflow_function(*args, **kwargs)

    def _validate_params(self, params: dict):
        input_schema = dict(signature(self.workflow_function).parameters)
        # Convert to the format that create_model expects
        schema_dict = {}
        for k, v in input_schema.items():
            if v.annotation != v.empty:
                # If the parameter has a default value
                if v.default != v.empty:
                    schema_dict[k] = (v.annotation, v.default)
                else:
                    schema_dict[k] = (
                        v.annotation,
                        ...,
                    )  # Using ... as required (no default)
            else:
                # Handle parameters without type annotations
                if k in params:
                    schema_dict[k] = (type(params[k]), ...)
                else:
                    schema_dict[k] = (
                        str,
                        ...,
                    )  # Default to string if no type info available
        try:
            pydantic_model = create_model("DynamicModel", **schema_dict)
            pydantic_model(**params)
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            print("Does your workflow function have type annotations?")
            return False


class Trial:
    """
    WIP
    A class to run a workflow and evaluate its performance.
    This takes function params, but also a config dict which might have configs like the following:
    - model
    - temperature
    """

    def __init__(self, workflow: Workflow, default_args: Optional[dict]):
        """ """
        self.workflow = workflow
        self.default_args = default_args
        self.params: Optional[dict] = None

    def run(self, kwargs: Optional[dict] = None):
        """
        Run the workflow and return the result.
        """
        if kwargs is None:
            kwargs = self.default_args
        result = self._run(self.workflow, kwargs)
        return (result,)

    def _run(self, kwargs: Optional[dict] = None):
        """
        Run the workflow and return the result.
        """
        if kwargs is None:
            kwargs = self.default_args
        result = self.workflow(**kwargs)
        return result


class Harness:
    pass


class Orchestrator:
    pass


def evaluate(curation):
    """
    Boilerplate for evaluation function.
    """
    score = 4.5
    return score


if __name__ == "__main__":
    # certs = get_all_certs()
    # topic = "Business Strategy"
    # curation = Mentor(topic, cache=True)
    # console.print(curation)
    # score = evaluate(curation)
    # console.print(f"[bold green]Score: {score}[/]")
    w = Workflow(
        Mentor,
        {"topic": "Business Strategy", "cache": True, "return_curriculum": False},
    )
    w("Business Strategy")
