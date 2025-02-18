from Chain import Model
from Kramer.courses.LearningPath import (
    description_chain,
    build_Curation_from_string,
    Curation,
)
from time import time
import json

example_curation = build_Curation_from_string(
    """
Data Science Professional Certificate by KNIME
Data Science Foundations: Fundamentals
Low Code/No-Code Data Literacy with KNIME: From Basic to Advanced
Introduction to Artificial Intelligence
Machine Learning and AI Foundations: Classification Modeling
Generative AI: Introduction to Large Language Models
The Non-Technical Skills of Effective Data Scientists
"""
)


def test_model(curation: Curation, model_name: str) -> dict:
    """
    Run the example chain for a model and return a dict.
    model_name: name of the model
    status: SUCCESS, FAIL
    output: the output of the model
    cold_boot: duration of task on first time
    warm_boot: duration of task on second time
    """
    try:
        start = time()
        output = description_chain(curation, preferred_model=model_name)
        cold_boot = time() - start

        start = time()
        output = description_chain(curation, preferred_model=model_name)
        warm_boot = time() - start
        return {
            "model_name": model_name,
            "status": "SUCCESS",
            "output": output,
            "cold_boot": cold_boot,
            "warm_boot": warm_boot,
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "status": "FAIL",
            "output": str(e),
            "cold_boot": None,
            "warm_boot": None,
        }


if __name__ == "__main__":
    results = {}
    models = [model for model in Model.models["ollama"] if "deepseek" not in model]
    for index, model in enumerate(models):
        print(f"Testing model #{index+1}: {model}")
        result = test_model(example_curation, model)
        print(json.dumps(result, indent=2))
        with open("model_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
