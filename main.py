# Git Repo which follow for build pipeline of document process.
# https://github.com/krishnaik06/RAG-Tutorials/blob/main/src/data_loader.py
# import os
#
# from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings
# import uuid
# from typing import List, Dict, Any,Tuple
# from sklearn.metrics.pairwise import cosine_similarity
# from pathlib import Path
#
# from sympy import public
#
#
# def process_all_pdf(pdf_dir):
#     all_document =[]
#     pdf_dir = Path(pdf_dir)
#     print(pdf_dir)
#     pdf_files =list(pdf_dir.glob("*.pdf"))
#     print(f"Found {len(pdf_files)} PDF Files to process.")
#
#     for pdf_file in pdf_files:
#         print(f"\n Processing: {pdf_file}")
#         try:
#             loader = PyPDFLoader(str(pdf_file))
#             documents = loader.load()
#
#             for doc in documents:
#                 doc.metadata['sourcefile'] = pdf_file.name
#                 doc.metadata['file_type'] = 'pdf'
#
#             all_document.extend(documents)
#             print(f"✔ Loaded {len(documents)} pages.")
#
#         except Exception as e:
#             print(f"{e}")
#
#     print(f" ✔ Total document loaded:{len(all_document)}")
#     return all_document
#
#
# def split_documents(documents,chunk_size=1000,chunk_overlap=200):
#     text_spliter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n","\n"," ",""]
#     )
#     split_docs = text_spliter.split_documents(documents)
#     print(f" ✔ Split {len(documents)} documents into {len(split_docs)} chunks.")
#
#     return split_docs
#
# class EmbeddingManager:
#     def __init__(self):
#         self.model = None
#         self.model_name =str("all-MiniLM-L6-v2")
#         self.load_model()
#
#     def __int__(self,model_name:str ="all-MiniLM-L6-v2"):
#         self.model_name=model_name
#         self.model = None
#         self.load_model()
#
#     def load_model(self):
#         try:
#             print(f"Loading embedding model: {self.model_name}")
#             self.model = SentenceTransformer(self.model_name)
#             print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
#         except Exception as e:
#             print(f"Error loading model {self.model_name}: {e}")
#             raise
#     def generate_embeddings(self,texts:List[str])->np.ndarray:
#         if not self.model:
#             raise ValueError("Model not loaded")
#
#         print(f"Generating embeddings for {len(texts)} texts...")
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         print(f"Generated embeddings with shape: {embeddings.shape}")
#         return embeddings
#
#
# class VectorStore:
#     def __init__(self):
#         self.collection_name = str("pdf_documents")
#         self.persist_directory = str("./data/vector_store")
#         self.client = None
#         self.collection = None
#         self._initialize_store()
#
#     def _initialize_store(self):
#         try:
#             os.makedirs(self.persist_directory,exist_ok=True)
#             self.client = chromadb.PersistentClient(path=self.persist_directory)
#
#             # Get or create collection
#             self.collection = self.client.get_or_create_collection(
#                 name=self.collection_name,
#                 metadata={"description":"Pdf document embedding for RAG."}
#             )
#             print(f"Vector store initialized. Collection: {self.collection_name}")
#             print(f"Existing documents in collection: {self.collection.count()}")
#         except Exception as e:
#             print(f"Error initializing vector store: {e}")
#             raise
#     def add_documents(self,documents:List[Any],embeddings:np.ndarray):
#         if len(documents) != len(embeddings):
#             raise ValueError("Number of documents must match number of embedding.")
#
#         print(f"Adding {len(documents)} documents to vector store...")
#
#         # Prepare data for chromDB
#         ids = []
#         metadatas = []
#         documents_text = []
#         embeddings_list = []
#
#         for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
#             doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
#             ids.append(doc_id)
#
#             # prepare metadata
#             metadata = dict(doc.metadata)
#             metadata['doc_index'] = i
#             metadata['content_length'] = len(doc.page_content)
#             metadatas.append(metadata)
#
#             documents_text.append(doc.page_content)
#             embeddings_list.append(embedding.tolist())
#
#             try:
#                 self.collection.add(
#                     ids=ids,
#                     embeddings=embeddings_list,
#                     metadatas=metadatas,
#                     documents=documents_text
#                 )
#                 print(f"Successfully added {len(documents)} documents to vector store")
#                 print(f"Total documents in collection: {self.collection.count()}")
#             except Exception as e:
#                 print(f"Error adding documents to vector store: {e}")
#                 raise
#
# def main():
#     # all_docs=process_all_pdf("./data")
#     # doc_chunks =split_documents(all_docs)
#     # print(doc_chunks)
#     # embedding=EmbeddingManager()
#     # embedding.load_model()
#     # embedding.generate_embeddings(["sadas","asdas"])
#     # print(embedding)
#     storage = VectorStore()

# if __name__ == '__main__':
#     main()


from rag_pipeline.data_loader import load_all_documents
from rag_pipeline.vectorstore import DbVectorStore
from rag_pipeline.search import RAGSearch

# Example usage
if __name__ == "__main__":
    # docs = load_all_documents("data")
    # print(docs)
    store = DbVectorStore("personal_store")
    store.build_from_documents(docs)
    # store.load()
    # print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "cat is brown."
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)