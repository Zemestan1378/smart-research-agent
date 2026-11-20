
import ast
import math
import operator

from app.rag.retriever import get_retriever
from app.rag.vector_store import get_vector_store
from app.llm.ollama_client import get_ollama_client


# --------------------------------------------------
# TOOL 1: Search Documents
# --------------------------------------------------

def search_documents(
    query: str,
    top_k: int = 4,
) -> dict:
    """Search uploaded documents using semantic search."""

    retriever = get_retriever()

    results = retriever.search(
        query=query,
        top_k=top_k,
    )

    return {
        "results_count": len(results),
        "results": [
            {
                "text": item["text"],
                "source": item["metadata"].get("source"),
                "page": item["metadata"].get("page"),
                "distance": float(item["distance"]),
            }
            for item in results
        ],
    }


# --------------------------------------------------
# TOOL 2: Summarize Document
# --------------------------------------------------

def summarize_document(filename: str | None = None) -> dict:
    """
    Summarize a document.
    If filename is omitted, summarize all indexed documents.
    """

    vector_store = get_vector_store()
    collection = vector_store.collection

    if filename:
        stored_data = collection.get(
            where={"source": filename},
            include=["documents", "metadatas"],
        )
    else:
        stored_data = collection.get(
            include=["documents", "metadatas"],
        )

    documents = stored_data.get("documents") or []
    metadatas = stored_data.get("metadatas") or []

    if not documents:
        return {
            "error": "No indexed documents were found."
        }

    grouped_documents = {}

    for text, metadata in zip(documents, metadatas):
        source = metadata.get("source", "unknown")

        grouped_documents.setdefault(source, []).append({
            "text": text,
            "page": metadata.get("page", 0),
            "chunk_index": metadata.get("chunk_index", 0),
        })

    llm = get_ollama_client()
    summaries = []

    for source, chunks in grouped_documents.items():
        chunks.sort(
            key=lambda item: (
                item["page"],
                item["chunk_index"],
            )
        )

        content = "\n\n".join(
            chunk["text"] for chunk in chunks
        )

        # Limit context size for the local LLM
        content = content[:12000]

        summary = llm.generate(
            prompt=f"""
Summarize the following document.

DOCUMENT:
{content}

Provide:
1. Main topic
2. Important points
3. Key conclusions
""",
            system_prompt=(
                "You are a document summarization assistant. "
                "Use only the supplied document content. "
                "Do not invent information."
            ),
        )

        summaries.append({
            "filename": source,
            "summary": summary,
        })

    return {
        "documents_summarized": len(summaries),
        "summaries": summaries,
    }


# --------------------------------------------------
# TOOL 3: Safe Calculator
# --------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_math(node):
    """Evaluate arithmetic expressions without using eval()."""

    if isinstance(node, ast.Expression):
        return _evaluate_math(node.body)

    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric values are allowed.")

        return node.value

    if isinstance(node, ast.BinOp):
        operator_function = _ALLOWED_OPERATORS.get(
            type(node.op)
        )

        if operator_function is None:
            raise ValueError("Unsupported mathematical operator.")

        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        result = operator_function(left, right)

        if not isinstance(result, (int, float)):
            raise ValueError("Invalid calculation result.")

        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Calculation result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        operator_function = _ALLOWED_UNARY_OPERATORS.get(
            type(node.op)
        )

        if operator_function is None:
            raise ValueError("Unsupported unary operator.")

        return operator_function(
            _evaluate_math(node.operand)
        )

    raise ValueError("Invalid mathematical expression.")


def calculate(expression: str) -> dict:
    """Calculate a basic mathematical expression safely."""

    if not expression or len(expression) > 100:
        raise ValueError("Invalid expression length.")

    try:
        parsed_expression = ast.parse(
            expression,
            mode="eval",
        )

        result = _evaluate_math(parsed_expression)

        return {
            "expression": expression,
            "result": result,
        }

    except (SyntaxError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(
            f"Calculation failed: {str(error)}"
        ) from error
