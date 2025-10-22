### Purpose of this directory
- To experiment with other embedding models like Jina, Stella, Voyager, Qwen, etc.
- To be able to evaluate their performance on my test case (Curator script), along with different rerankers
- This will eventally be its own project + plugabble / swappable in my other scripts as part of broader RAG tooling

NOTE: I've done preliminary testing of reranker models in Curator project; combine these at a later date.

### Considerations
- define your quality metrics / loss functions so that when you swap embedding models and rerankers you can evaluate their performance
- consider different types of models for different tasks (sentence > passage, sentence > passage)

Inspired by: https://www.datastax.com/blog/best-embedding-models-information-retrieval-2025

### Evaluation
#### Test data set

Query (str) and matches (list of five course titles).

```json
{"query": ..., "matches": [...]}
```

#### Evaluation logic

Score is the number of generated matches that are found in the test matches.

```pseudocode
for model in models:
    for datum in data: # list of those json objects as dictionaries
        query = datum['query']
        test_matches = datum['matches']
        generated_matches = chroma_query(query, model)
        score = 0
        for match in generated_matches:
            if match in test_matches:
                score += 1
```
