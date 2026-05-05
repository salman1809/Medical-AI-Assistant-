Engineered a scalable RAG pipeline indexing 500+ medical documents with chunking, embedding, and ChromaDB vector search, 
achieving 87% answer relevance on a 200-query test set.
Optimized retrieval parameters (chunk size 512, overlap 64, k=5), reducing irrelevant context noise by ~30% compared to baseline 
configuration.
Designed structured chain-of-thought prompts that improved response accuracy by 22% over zero-shot prompting on a held-out 
evaluation set.
Deployed a Streamlit app supporting real-time Q&A; average query response time under 2 seconds on CPU inference
