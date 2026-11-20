
from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text.strip(),
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                    "file_type": "pdf",
                },
            })

    return pages


def load_txt(file_path: str) -> list[dict]:
    """
    Extract text from a TXT file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("The provided file is not a TXT file.")

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "metadata": {
            "source": path.name,
            "page": 1,
            "file_type": "txt",
        },
    }]


def load_document(file_path: str) -> list[dict]:
    """
    Select the appropriate loader based on file extension.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    raise ValueError(f"Unsupported file type: {extension}")
