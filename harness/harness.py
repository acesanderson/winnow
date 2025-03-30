"""
MVP'ing my way to an evaluation framework.
"""

from Mentor import Mentor
from Kramer import get_all_certs
from Chain import Chain, MessageStore
from rich.console import Console
from typing import Callable, Optional
from pydantic import BaseModel
from inspect import signature, Signature
from types import MappingProxyType

Chain._message_store = MessageStore()
console = Console()


class Workflow:
    """
    This defines a workflow: takes an input and returns an output.
    """

    def __init__(self, workflow_function: Callable):
        """
        Initialize the workflow with the given function.
        """
        if not callable(workflow_function):
            raise TypeError("Workflow function must be callable.")
        self.name = workflow_function.__name__
        self.description = workflow_function.__doc__ or ""
        self.params = dict(signature(workflow_function).parameters)
        self.params = {
            k: "bool" if v.annotation.__name__ == "_empty" else v.annotation.__name__
            for k, v in self.params.items()
        }  # The "_empty" is because values I set to False get interpreted as None sans the type hint
        self.workflow_function = workflow_function

    def __call__(self, *args, **kwargs):
        """
        Call the workflow with the given arguments.
        """
        return self.workflow_function(*args, **kwargs)


class Harness:
    """
    A class to run a workflow and evaluate its performance.
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
    w = Workflow(Mentor)
    w("Business Strategy")
