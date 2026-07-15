import os
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    PyMuPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq




def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs




class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading Embedding Model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded. Cannot generate embeddings.")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


\

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_dictionary: str = "../data/vector_store"):
        self.collection_name = collection_name
        self.persist_dictionary = persist_dictionary
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_dictionary, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dictionary)

            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Collection of PDF document embeddings", "hnsw:space": "cosine"}
            )
            print(f"Vector store initialized with collection: {self.collection_name}")
            print(f"Existing documents in store: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        document_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_text,
                embeddings=embeddings_list
            )
            print(f"Successfully added {len(documents)} documents to vector store.")
            print(f"Total documents in store after addition: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise




class RAGRetreiver:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        allowed_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}, Allowed domains: {allowed_domains}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # allowed_domains=None or [] means unrestricted (e.g. admin role).
        # Otherwise restrict Chroma's search to only chunks tagged with one
        # of the caller's allowed domains, at the DB level -- not filtered
        # after the fact, so out-of-domain chunks never even surface.
        where_clause = None
        if allowed_domains:
            if len(allowed_domains) == 1:
                where_clause = {"domain": allowed_domains[0]}
            else:
                where_clause = {"domain": {"$in": allowed_domains}}

        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause

        try:
            results = self.vector_store.collection.query(**query_kwargs)

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents after applying score threshold.")
            else:
                print("No documents retrieved from vector store.")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []



class LLMManager:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("GROQ_API_KEY_RAG")
        if not resolved_key:
            raise ValueError(
                "No Groq API key found.\n"
                "Either pass api_key=... or set the GROQ_API_KEY environment variable.\n"
                "Get a free key at https://console.groq.com"
            )

        self.client = Groq(api_key=resolved_key)
        print(f"LLMManager ready  →  model: {self.model_name}")

    def _build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        max_context_chars: int = 4000,
    ) -> str:
        context_parts = []
        total_chars = 0

        for rank, doc in enumerate(retrieved_docs, 1):
            snippet = doc["content"].strip()
            header = (
                f"[Source {rank} | "
                f"score={doc.get('similarity_score', 0):.3f} | "
                f"file={doc.get('metadata', {}).get('source', 'unknown')}]"
            )
            block = f"{header}\n{snippet}"

            if total_chars + len(block) > max_context_chars:
                break
            context_parts.append(block)
            total_chars += len(block)

        context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No context retrieved."

        return (
            f"Use ONLY the context below to answer the question.\n"
            f"If the answer is not in the context, say \"I don't have enough information.\"\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}"
        )

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found. Please try a different query.",
                "model": self.model_name,
                "sources": [],
                "num_docs": 0,
            }

        user_prompt = self._build_prompt(query, retrieved_docs)

        system_prompt = (
            "You are a precise, helpful assistant. "
            "Answer questions strictly based on the provided context. "
            "Be concise and factual. "
            "Cite source numbers (e.g. [Source 1]) when referencing specific content."
        )

        print(f"Sending query to {self.model_name} …")

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content.strip()
        sources = list({
            doc.get("metadata", {}).get("source", "unknown")
            for doc in retrieved_docs
        })

        print(f"Answer generated  ({len(answer)} chars, {len(retrieved_docs)} chunks used).")

        return {
            "answer": answer,
            "model": self.model_name,
            "sources": sources,
            "num_docs": len(retrieved_docs),
        }



class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm_manager,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.top_k = top_k
        self.score_threshold = score_threshold

    def ask(self, query: str, allowed_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        print("=" * 60)
        print(f"QUERY : {query}")
        print("=" * 60)

        retrieved_docs = self.retriever.retrieve(
            query,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            allowed_domains=allowed_domains,
        )

        result = self.llm_manager.generate(query, retrieved_docs)
        result["retrieved_docs"] = retrieved_docs

        print("\n📄 ANSWER:")
        print("-" * 40)
        print(result["answer"])
        print("-" * 40)
        print(f"📚 Sources : {result['sources']}")
        print(f"🔢 Chunks used : {result['num_docs']}")
        print("=" * 60)

        return result




embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
rag_retriever = RAGRetreiver(vectorstore, embedding_manager)



def ingest_pdf(filepath: str, domain: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    # Check if this exact file source already exists in the Chroma collection
    existing = vectorstore.collection.get(where={"source": filepath})
    if existing and len(existing.get("ids", [])) > 0:
        print(f"Skipping ingestion: {filepath} is already in the database.")
        return {
            "status": "skipped", 
            "message": "File already ingested", 
            "filepath": filepath
        }

    try:
        loader = PyMuPDFLoader(filepath)
        pdf_documents = loader.load()

        chunks = split_documents(pdf_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Tag every chunk with its domain BEFORE embedding/storage.
        # VectorStore.add_documents() copies doc.metadata verbatim, so this
        # is the only place domain tagging needs to happen.
        for chunk in chunks:
            chunk.metadata["domain"] = domain

        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        vectorstore.add_documents(chunks, embeddings)

        return {
            "status": "success",
            "filepath": filepath,
            "domain": domain,
            "num_chunks": len(chunks),
        }
    except Exception as e:
        print(f"Error ingesting PDF '{filepath}': {e}")
        return {
            "status": "error",
            "filepath": filepath,
            "error": str(e),
        }


def retrieve_from_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    allowed_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return rag_retriever.retrieve(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        allowed_domains=allowed_domains,
    )