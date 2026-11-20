
from app.rag.retriever import get_retriever
from app.llm.ollama_client import get_ollama_client


SYSTEM_PROMPT = """
You are a careful research assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not available in the context, say:
   "I could not find the answer in the provided documents."
3. Do not invent facts.
4. Cite the source filename and page number when possible.
5. Be clear and concise.
"""


class RAGQuestionAnswering:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_ollama_client()

    def answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Retrieve relevant chunks
        chunks = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        # Do not call the LLM when no documents exist
        if not chunks:
            return {
                "question": question,
                "answer": (
                    "I could not find the answer "
                    "in the provided documents."
                ),
                "sources": [],
            }

        # Build context with source information
        context_parts = []
        sources = []

        for chunk in chunks:
            metadata = chunk["metadata"]

            source = {
                "filename": metadata.get("source"),
                "page": metadata.get("page"),
            }

            context_parts.append(
                f"""
Source: {source['filename']}
Page: {source['page']}

Content:
{chunk['text']}
"""
            )

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        prompt = f"""
Use the following document context to answer the question.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Provide an accurate answer with source citations.
"""

        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


_qa_engine = None


def get_qa_engine() -> RAGQuestionAnswering:
    global _qa_engine

    if _qa_engine is None:
        _qa_engine = RAGQuestionAnswering()

    return _qa_engine
