"""
FAISS Retriever - RAG Retrieval Stage

This module handles the retrieval part of RAG:
- Loads the FAISS index we built in ingest.py
- Takes a user question
- Finds the most relevant chunks (semantic search)
- Returns chunks with similarity scores

RAG Flow:
1. User asks: "What skills are required?"
2. This module searches FAISS for relevant chunks
3. Returns top-k chunks (e.g., top 5)
4. These chunks are sent to Ollama with the question
5. Ollama generates answer based on retrieved context

Semantic Search:
- Finds chunks by meaning, not just keywords
- Query "programming languages" finds chunks about "Python", "Java", etc.
- Uses embeddings to find similar text in vector space
"""

import os
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy import FAISS to avoid DLL loading issues on Windows
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    FAISS_ERROR = str(e)


class FAISSRetriever:
    """Retrieves relevant chunks from FAISS index using semantic search."""
    
    def __init__(self, index_path: str = "db/faiss.index", chunks_path: str = "db/chunks.pkl", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks metadata pickle
            model_name: Embedding model (MUST match the one used in ingest.py)
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                f"Run: python -m rag.ingest"
            )
        
        if not FAISS_AVAILABLE:
            error_msg = f"""
            ❌ FAISS is not available!
            
            Error: {FAISS_ERROR}
            
            This is often caused by insufficient virtual memory (paging file) on Windows.
            
            Solutions:
            1. Increase Windows paging file size:
               - Press Win+R, type: sysdm.cpl
               - Advanced → Performance Settings → Advanced → Virtual Memory
               - Set to "System managed size" or increase manually
            
            2. Reinstall FAISS:
               pip uninstall faiss-cpu
               pip install faiss-cpu --no-cache-dir
            
            3. Restart your computer after changing paging file settings
            """
            raise ImportError(error_msg)
        
        print(f"Loading FAISS index from {index_path}...")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"[OK] Loaded {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5, include_files: List[str] = None, 
                 exclude_files: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query with optional file filtering.
        
        How it works:
        1. Convert query to embedding (same model as ingestion)
        2. Search FAISS for closest embeddings
        3. Filter by source_file if include_files/exclude_files specified
        4. Return chunks with similarity scores
        
        Args:
            query: User question or search text
            top_k: Number of chunks to retrieve
            include_files: List of source_file names to include (e.g., ['resume.pdf'])
                          If None, all files are included (unless excluded)
            exclude_files: List of source_file names to exclude (e.g., ['resume.pdf'])
                          If None, no files are excluded
            
        Returns:
            List of chunk dicts with 'text', 'source_file', 'page', 'score', 'chunk_id'
        """
        # Encode query to embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        query_embedding = query_embedding.astype('float32')
        
        # Search for more chunks if filtering is needed (to ensure we get top_k after filtering)
        search_k = top_k * 3 if (include_files or exclude_files) else top_k
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(search_k, len(self.chunks)))
        
        # Retrieve chunks with scores and apply filtering
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                
                # Apply file filtering
                source_file = chunk.get('source_file', chunk.get('source', ''))
                
                # Check include_files filter
                if include_files is not None:
                    if source_file not in include_files:
                        continue
                
                # Check exclude_files filter
                if exclude_files is not None:
                    if source_file in exclude_files:
                        continue
                
                # Convert distance to similarity score (0-1, higher = more similar)
                similarity = 1 - (dist / 2.0)  # Normalized L2 distance → similarity
                chunk['score'] = float(similarity)
                chunk['chunk_id'] = int(idx)
                results.append(chunk)
                
                # Stop when we have enough filtered results
                if len(results) >= top_k:
                    break
        
        return results


if __name__ == "__main__":
    # Test retrieval
    retriever = FAISSRetriever()
    query = "What skills are required?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant chunks:\n")
    for i, chunk in enumerate(results, 1):
        print(f"Chunk {i} (score: {chunk['score']:.3f}, from {chunk['source_file']}, page {chunk['page']}):")
        print(f"  {chunk['text'][:200]}...\n")

