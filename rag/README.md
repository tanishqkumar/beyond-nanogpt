## RAG (Retrieval-Augmented Generation)

**Dependencies:** `pip install langchain dspy-ai faiss-cpu sentence-transformers`

This directly implements the fundamentals of retrieval-augmented generation (RAG), a method to 
give LLMs access to knowledge on the fly so they produce more knowledgeable, detailed, and factual answers. 
Important on its own when LLMs are treated as knowledge bases/search engines, but also a key component of 
modern frameworks seeking to build agentic/autonomous systems around base language models. 

- [ ] Train Small Embedding and Reranking Models `rag/train_embedding.py`
  Implements training of small sentence embedding models from scratch using contrastive learning on text pairs. Shows how to create positive/negative pairs from documents, implement InfoNCE loss, and evaluate on semantic similarity tasks. Core challenge: balancing embedding dimensionality with retrieval quality.

- [ ] Hello Docs Q&A `rag/hello_docs.py`
  Gives a tiny LM access to a local folder of Markdown/PDF notes to answer arbitrary questions. Covers document chunking strategies, embedding generation, vector search with FAISS, and prompt-stuffing with retrieved context. Core challenge: choosing chunk sizes and relevance scores so the model's limited context window isn't wasted on noise.

- [ ] Multi-Hop Decomposition RAG `rag/multi_hop.py`
  Uses the LM to decompose a complex question into sub-questions, retrieves evidence for each step, and then aggregates the answers—illustrating iterative reasoning + retrieval loops. Shows how to persist intermediate state and avoid retrieval loops that blow up token counts.

- [ ] Self-Reranking Feedback Loop `rag/self_reranking.py`
  Implements a feedback mechanism where the LM scores and reranks its own retrieved documents based on relevance to the query. The model learns to critique its own retrieval choices and iteratively improve context selection, demonstrating self-reflective RAG architectures.

- [ ] Sparse and Dense Retrieval `rag/hybrid_retrieval.py`
  Combines traditional BM25 sparse retrieval with dense vector search using learned embeddings. Shows how to implement reciprocal rank fusion to merge results from both approaches, highlighting the complementary strengths of lexical and semantic matching.

- [ ] Graph RAG `rag/graph_rag.py`
  Builds a knowledge graph from documents and uses graph traversal alongside traditional vector search for retrieval. Implements entity extraction, relation modeling, and graph-based query expansion to capture complex multi-entity relationships that pure vector search might miss.