"""
PDF Ingestion - Building the RAG Knowledge Base

What is RAG?
- RAG = Retrieval-Augmented Generation
- Step 1 (this file): Build a searchable index from your PDFs
- Step 2 (retrieve.py): Search the index to find relevant chunks
- Step 3 (job_agent.py): Send chunks + question to Ollama for answer

What are Embeddings?
- Embeddings are vector representations of text (arrays of numbers)
- Similar text has similar embeddings (close in vector space)
- Example: "Python programming" and "coding in Python" have similar embeddings
- We use SentenceTransformers to convert text → embeddings

What is FAISS?
- FAISS = Facebook AI Similarity Search
- Fast vector database for finding similar embeddings
- Perfect for CPU-only setups (no GPU needed)
- Stores embeddings so we can quickly find relevant chunks

This script:
1. Loads all PDFs from /data folder
2. Extracts text and chunks it into smaller pieces
3. Creates embeddings using SentenceTransformers
4. Stores embeddings in FAISS index (db/faiss.index)
5. Stores chunk text + metadata in db/chunks.pkl
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import pypdf
import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy import FAISS to avoid DLL loading issues on Windows
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    FAISS_ERROR = str(e)


class PDFIngester:
    """Handles PDF ingestion: extract → chunk → embed → index"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500):
        """
        Initialize ingester.
        
        Args:
            model_name: SentenceTransformer model (CPU-friendly, small, fast)
            chunk_size: Max characters per chunk (smaller = more precise retrieval)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        print(f"[OK] Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page numbers.
        
        Returns:
            List of dicts: [{'text': '...', 'page': 1}, ...]
        """
        pages = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        'text': text,
                        'page': page_num
                    })
        return pages
    
    def chunk_text(self, text: str, source_file: str, page: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for better retrieval.
        
        Why chunking?
        - Long documents are split into smaller pieces
        - Each chunk becomes a searchable unit
        - Better precision: retrieve only relevant sections
        
        Returns:
            List of chunk dicts with metadata
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is too long, split by sentences
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source_file': source_file,
                        'page': page
                    })
                    current_chunk = ""
                
                # Split long paragraph
                sentences = para.replace('. ', '.\n').split('\n')
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    
                    if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                        current_chunk += " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'source_file': source_file,
                                'page': page
                            })
                        current_chunk = sent
            else:
                # Add paragraph to current chunk if it fits
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'source_file': source_file,
                            'page': page
                        })
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source_file': source_file,
                'page': page
            })
        
        return chunks
    
    def build_index(self, data_dir: str = "data", db_dir: str = "db"):
        """
        Main function: build FAISS index from all PDFs in /data.
        
        Process:
        1. Find all PDFs in data_dir
        2. Extract and chunk text
        3. Generate embeddings
        4. Save to FAISS index + metadata pickle
        
        Args:
            data_dir: Folder containing PDFs
            db_dir: Folder to save index files
        """
        data_path = Path(data_dir)
        pdf_paths = list(data_path.glob("*.pdf"))
        
        if not pdf_paths:
            raise ValueError(f"No PDFs found in {data_dir}/")
        
        print(f"\nFound {len(pdf_paths)} PDF(s):")
        for pdf in pdf_paths:
            print(f"  - {pdf.name}")
        
        # Step 1: Extract and chunk all PDFs
        all_chunks = []
        chunks_per_pdf = {}  # Track chunks per PDF for summary
        chunk_id_counter = 0  # Unique ID counter
        
        for pdf_path in pdf_paths:
            print(f"\nProcessing: {pdf_path.name}...")
            pages = self.extract_text_from_pdf(str(pdf_path))
            source_file = pdf_path.name
            pdf_chunk_count = 0
            
            for page_data in pages:
                chunks = self.chunk_text(page_data['text'], source_file, page_data['page'])
                # Add unique chunk_id to each chunk
                for chunk in chunks:
                    chunk['chunk_id'] = chunk_id_counter
                    chunk_id_counter += 1
                    pdf_chunk_count += 1
                all_chunks.extend(chunks)
            
            chunks_per_pdf[source_file] = pdf_chunk_count
        
        print(f"\n[OK] Total chunks created: {len(all_chunks)}")
        
        # Step 2: Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Step 3: Create FAISS index
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
        
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Step 4: Save index and metadata
        os.makedirs(db_dir, exist_ok=True)
        index_path = os.path.join(db_dir, "faiss.index")
        chunks_path = os.path.join(db_dir, "chunks.pkl")
        
        faiss.write_index(index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        
        print(f"\n[OK] Index saved to: {index_path}")
        print(f"[OK] Chunks saved to: {chunks_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("INDEXING SUMMARY")
        print("="*60)
        print(f"Total PDFs processed: {len(pdf_paths)}")
        print(f"Total chunks indexed: {len(all_chunks)}")
        print("\nChunks per PDF:")
        for pdf_name, chunk_count in chunks_per_pdf.items():
            print(f"  - {pdf_name}: {chunk_count} chunks")
        print("="*60)


if __name__ == "__main__":
    # Run ingestion
    ingester = PDFIngester()
    ingester.build_index()

