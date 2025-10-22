from conduit.sync import Model
from kramer.courses.LearningPath import (
    description_conduit,
    Curation,
)
from kramer.database.MongoDB_certs import get_all_certs
from time import time
import json

# Subset of models that I can actually run.
models = [
    "gemma2:27b",
    "gemma2:latest",
    "mistral-small:latest",
    "mistral-nemo:latest",
    "mistral:latest",
    "mixtral:8x7b",
    "llama3.2:latest",
    "llama3.1:latest",
    "llama3.2-vision:latest",
    "llama3.3:latest",
    "llama3.1:70b",
    "phi3.5:latest",
    "phi4:latest",
    "qwq:latest",
    "qwen2.5:14b",
    "qwen2.5:latest",
]


def test_model(curation: Curation, model_name: str) -> dict:
    """
    Run the example conduit for a model and return a dict.
    model_name: name of the model
    status: SUCCESS, FAIL
    output: the output of the model
    cold_boot: duration of task on first time
    warm_boot: duration of task on second time
    """
    try:
        start = time()
        output = description_conduit(curation, preferred_model=model_name)
        cold_boot = time() - start

        start = time()
        output = description_conduit(curation, preferred_model=model_name)
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


def test_all_models(curation: Curation):
    """
    Run the curation through models.
    """
    for index, model in enumerate(models):
        print(f"Testing model #{index + 1}: {model}")
        result = test_model(curation, model)
        print(json.dumps(result, indent=2))
        # Save results
        with open("model_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    certs = get_all_certs()
    for index, cert in enumerate(certs):
        print(f"Processing #{index + 1} of {len(certs)}: {cert.title}")
        test_all_models(cert)
