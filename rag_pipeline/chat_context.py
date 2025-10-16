

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sympy.polys.polyconfig import query

from rag_pipeline.vectorstore import DbVectorStore


class ChatContext:
    def __init__(self, model_name="gemma3:1b"):
        """Initialize RAG search with local Ollama model"""
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            num_predict=512,  # Max tokens to generate
        )
        print(f"[INFO] Ollama LLM initialized: {model_name}")

    def search_and_summarize(self, query, context_docs):
            """Search and summarize with context"""

            # Create prompt with system and user messages
            if len(context_docs) == 0:
                messages = [
                    SystemMessage(content="You are a helpful AI assistant. Answer based on the Teacher way"),
                    HumanMessage(content=f"""Question: {query}
                     Answer the question like a teacher in a classroom.
                    Give answer if json formate like question and answer. Don't give description,to the point answer.""")
                ]
            else:
                # Build context from retrieved documents
                # Handle both dict and Document objects
                context_parts = []
                for doc in context_docs:
                    if isinstance(doc, dict):
                        # If it's a dict, try common keys
                        content = doc.get('page_content') or doc.get('content') or doc.get('text') or str(doc)
                        context_parts.append(content)
                    else:
                        # If it's a Document object
                        context_parts.append(doc.page_content)

                context = "\n\n".join(context_parts)

                messages = [
                    SystemMessage(content="You are a helpful AI assistant. Answer based on the provided context."),
                    HumanMessage(content=f"""Context:{context}   
                    Question: {query}
                    Answer the question based on the context above.
                    Give answer if json formate like question and answer. Don't give description,to the point answer. """)
                ]

            # Get response
            response = self.llm.invoke(messages)
            return response.content


# Usage
if __name__ == "__main__":
    rag = ChatContext()
    store=DbVectorStore("chat_store")
    store.load()
    docs=[]
    chache_docs=[]
    while True:
        query=input("Enter Query:")
        if query ==":q":
            break
        chache_docs=store.query(query,2)
        summary = rag.search_and_summarize(query, chache_docs)
        print(f"\n\n{summary}")
        doc = Document(
            page_content=summary,
            metadata={
                "query": query,
            }
        )
        chache_docs.append(doc)
        docs.append(doc)
    if len(docs) >0:
        store.build_from_documents(docs)
        store.save()
    print("👓 All things saved well.")