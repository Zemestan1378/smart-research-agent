
from pathlib import Path
from uuid import uuid4
from app.api.schemas import AgentRequest
from app.agent.agent import get_agent
from app.api.schemas import SearchRequest
from app.rag.retriever import get_retriever
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.api.schemas import QuestionRequest
from app.rag.qa import get_qa_engine
from app.core.config import get_settings
from app.rag.pipeline import get_document_pipeline


router = APIRouter(prefix="/api/v1", tags=["Documents"])

settings = get_settings()


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document and ingest it into ChromaDB.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    # Generate a unique stored filename
    unique_filename = f"{uuid4().hex}{extension}"

    file_path = settings.documents_path / unique_filename

    try:
        # Save uploaded file
        content = file.file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_path.write_bytes(content)

        # Process document
        pipeline = get_document_pipeline()

        result = pipeline.ingest_document(
            str(file_path)
        )

        return {
            "message": "Document uploaded and processed successfully.",
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}",
        ) from error

    finally:
        file.file.close()
@router.post("/documents/search")
def search_documents(request: SearchRequest):
    """
    Search for semantically relevant document chunks.
    """

    try:
        retriever = get_retriever()

        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
        )

        return {
            "query": request.query,
            "results_count": len(results),
            "results": results,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(error)}",
        ) from error


@router.post("/documents/ask")
def ask_question(request: QuestionRequest):
    """
    Answer a question using uploaded documents.
    """

    try:
        qa_engine = get_qa_engine()

        result = qa_engine.answer(
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    
    except Exception as error:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(error)}",
        ) from error

@router.post("/agent/run")
def run_agent(request: AgentRequest):
    """
    Run the research agent with the user's message.
    """

    try:
        agent = get_agent()

        result = agent.run(
            user_message=request.message,
        )

        return {
            "message": request.message,
            "agent_response": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(error)}",
        ) from error
