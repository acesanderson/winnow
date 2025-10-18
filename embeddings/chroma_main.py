"""
Adding embeddings to Chroma is straightforward:

```python
collection.add(
    documents = [],
    embeddings = [],
    ids = []
)
```
Loss function can be created around the MBA mapping:
https://docs.google.com/spreadsheets/d/193LQyEQ-ZWZGQFRlcBUbtn1UNAF2QYZc5kBCkIwTdBM/edit?gid=181142691#gid=181142691

Short list of models listed here:
https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
Leaderboard:
https://huggingface.co/spaces/mteb/leaderboard
Datastax post:
https://www.datastax.com/blog/best-embedding-models-information-retrieval-2025

CUDA is turned on in embedding declaration. (device = "cuda")

TODO:
- create test data set
- create my own embedding_function ABC class so that I can implement models that require custom parameters and prompts (like stells_en_400M_v5)
 - stella has the s2s and s2p prompts which would be interesting to test (my use case is arguably s2p, at least for course transcripts).
 - compare s2s on course descriptions vs. s2p for course transcripts
"""

import torch
import sys
import chromadb
import json
import pickle
from chromadb.utils import embedding_functions
from kramer.database.MongoDB_CRUD import get_all_courses_sync
from kramer.database.MongoDB_course_mapping import get_course_title

client = chromadb.HttpClient(host="localhost", port=8001)


def update_progress(current, total) -> None:
    """
    This takes the index and len(iter) of a for loop and creates a pretty
    progress bar.
    """
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    if current != total:
        percent = float(current) * 100 / total
        bar = (
            GREEN
            + "=" * int(percent)
            + RESET
            + YELLOW
            + "-" * (100 - int(percent))
            + RESET
        )
        print(
            f"\rProgress: |{bar}| {current} of {total} | {percent:.2f}% Complete",
            end="",
        )
        sys.stdout.flush()
    elif current == total:
        print("\rProgress: |" + "=" * 100 + f"| {current} of {total} | 100% Complete\n")


def generate_test_data() -> tuple[list, list]:
    courses = get_all_courses_sync()
    ids = [str(course.course_admin_id) for course in courses]
    documents = [course.course_transcript for course in courses]
    return ids, documents


def test_model(
    model_name: str, test_data: tuple[list, list], queries: list[str]
) -> list[dict]:
    """
    Test the embedding function on a small dataset with an ephemeral collection.

    Args:
        model_name: str - the name of the sententence transformers embedding model
        test_data: list[list, list] - list of lists (ids and documents)
        queries: list[str] - a list of queries to test

    Returns:
        results: list[result]
        result: dict - keys: query: str, match: list[str]
    """
    collection_name = (
        f"test_collection_{model_name}".replace(" ", "_")
        .replace(".", "_")
        .replace("/", "-")
    )
    if model_name == "default":
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name, device="cuda"
        )
    new_collection = True
    try:
        collection = client.get_collection(
            collection_name, embedding_function=embedding_function
        )
        ids = collection.get()["ids"]
        print(f"# of ids in db: {len(ids)}, # of actual ids: {len(test_data[0])}")
        if len(ids) == len(test_data[0]):
            print(f"Collection {collection_name} already exists.")
            new_collection = False
        else:
            client.delete_collection(collection_name)
    except:
        pass
    if new_collection:
        collection = client.create_collection(
            collection_name, embedding_function=embedding_function
        )
        ids, documents = test_data
        batch_size = 100
        # Batch em up and load 'em into the collection
        for i in range(0, len(ids), batch_size):
            update_progress(i + 1, len(ids))
            collection.add(
                ids=ids[i : i + batch_size], documents=documents[i : i + batch_size]
            )
    results = []
    for query in queries:
        query_results = collection.query(query_texts=[query])
        matches = query_results["ids"][0]
        result = {"query": query, "match": matches}
        results.append(result)
    return results


def add_to_chroma(
    collection: chromadb.Collection, ids, documents, embeddings=[], model="default"
):
    """
    WIP: Wrapper function that adds ids and documents to a Chroma collection.
    Note: chroma's default is all-MiniLM-L6-v2.
    """
    # Handle default case, also if user provides embeddings
    if model == "default" or "MiniLM-L6-v2" in model:
        kwargs = {"ids": ids, "documents": documents}
        if embeddings:
            kwargs.update({"embeddings": embeddings})
    else:
        embedding_function = embedding_functions[model]
        embeddings = embedding_function(documents)
        kwargs = {"ids": ids, "documents": documents, "embeddings": embeddings}


if __name__ == "__main__":
    # Check if cuda is available
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")
        sys.exit()
    test_data = generate_test_data()
    models = [
        "intfloat/e5-mistral-7b-instruct",
        "voyageai/voyage-3-m-exp",
        "jinaai/jina-embedding-s-en-v1",
        "jinaai/jina-embeddings-v2-base-en",
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        "mixedbread-ai/mxbai-embed-large-v1",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en",
        # "NovaSearch/stella_en_400M_v5",
        # "Linq-AI-Research/Linq-Embed-Mistral",
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-multilingual-mpnet-base-v2",
        "paraphrase-albert-small-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-MiniLM-L3-v2",
        "distiluse-base-multilingual-cased-v1",
        "distiluse-base-multilingual-cased-v2",
    ]
    queries = [
        "Kubernetes",
        "Deep Learning with Python",
        "Read critical economic reports:  Price Indices and Inflation, Aggregate Production and Labor Demand, Unemployment",
        "How to best influence and keep track of public policy",
        "Social Media Marketing",
        "Sales Management",
        "Improving Operations at a large enterprise",
        "Generative AI for Business Analysts",
        # Semantic Understanding & Synonyms
        "Mastering object-oriented programming",
        "Building web applications from scratch",
        "Number crunching with computers",
        # Domain-Specific Jargon & Concepts
        "CI/CD pipeline automation",
        "ACID compliance in databases",
        "Agile ceremonies and artifacts",
        # Indirect References
        "Building the next Facebook",
        "Writing like Shakespeare",
        "Making machines think",
        # Historical Figures & Their Domains
        "Understanding networks like Dijkstra",
        "Statistical thinking like Bayes",
        "Designing like Don Norman",
        # Conceptual Relationships
        "Secure coding for banks",
        "Video game physics",
        "Restaurant website optimization",
        # Ambiguous Queries
        "Working with tables",
        "Understanding patterns",
        "Cloud architecture",
        # Regional/Cultural Knowledge
        "European privacy laws",
        "Silicon Valley engineering practices",
        "Japanese manufacturing principles",
        # Temporal Context
        "Modern web development",
        "Legacy system maintenance",
        "Future of mobile apps",
    ]
    all_results = []
    output = ""
    for model in models:
        print(f"Testing model: {model}")
        print("======================================")
        try:
            results = test_model(model_name=model, test_data=test_data, queries=queries)
            for result in results:
                result.update({"model": model})
                print(f"{result['query']}")
                for match in result["match"]:
                    try:
                        print(f"\t{get_course_title(int(match))}")
                    except:
                        print(f"\tCouldn't retrieve course title for {match}.")
            all_results.append(results)
        except:
            print(f"Failed to test model: {model}")
            continue
    # Save all_results to a pickle
    with open("all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
