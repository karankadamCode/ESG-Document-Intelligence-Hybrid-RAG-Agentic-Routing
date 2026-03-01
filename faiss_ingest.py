"""
ingest_faiss.py

Purpose:
Production-grade ingestion pipeline for Delaware case study.
Loads one or many PDFs, extracts text per page, chunks intelligently, embeds chunks,
and persists a FAISS index to disk with rich metadata for traceability.

Key behaviors:
1) Multi-document ingestion from a directory or file path
2) Stable document identifiers using filename + content hash
3) Chunk-level metadata:
   doc_id, source, page, chunk_index, ingest_run_id
4) Safe validation and actionable error messages

Author:
Karan Kadam
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from settings import load_settings


def _utc_now_iso() -> str:
    """
    Returns:
        Current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    """
    Args:
        text: Input text.

    Returns:
        Short sha256 hex digest for stable IDs.
    """
    h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return h[:16]


def _list_pdf_files(path: str) -> List[str]:
    """
    Collect PDF files from a directory or return a single PDF file path.

    Args:
        path: File path or directory path.

    Returns:
        List of PDF file paths.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If no PDFs are found.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Input path must be a non-empty string")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path}")

    if os.path.isfile(path):
        if not path.lower().endswith(".pdf"):
            raise ValueError(f"Input file is not a PDF: {path}")
        return [path]

    pdfs: List[str] = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    if not pdfs:
        raise ValueError(f"No PDF files found under directory: {path}")

    pdfs.sort()
    return pdfs


def _compute_doc_id(pdf_path: str) -> str:
    """
    Create a stable doc_id for a PDF.

    Strategy:
    doc_id = "<filename>|<sha16(file_bytes)>"

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Stable doc_id string.
    """
    base = os.path.basename(pdf_path)
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()
        digest = hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        digest = _sha256_text(pdf_path)
    return f"{base}|{digest}"


def load_pdf_pages(pdf_path: str, doc_id: str) -> List[Document]:
    """
    Load a PDF into page-level Documents.

    Args:
        pdf_path: PDF file path.
        doc_id: Stable document identifier.

    Returns:
        List of Documents, typically one per page.

    Raises:
        RuntimeError: If PDF extraction fails.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {pdf_path}") from e

    for d in pages:
        d.metadata = dict(d.metadata or {})
        d.metadata["doc_id"] = doc_id
        d.metadata["source"] = os.path.basename(pdf_path)

    return pages


def chunk_documents(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    ingest_run_id: str,
) -> List[Document]:
    """
    Split documents into overlapping chunks while preserving metadata.

    Args:
        docs: Page-level documents.
        chunk_size: Target chunk size in characters (LangChain splitter uses characters).
        chunk_overlap: Overlap in characters.
        ingest_run_id: Run identifier stored in metadata.

    Returns:
        Chunked documents.

    Raises:
        ValueError: If docs is empty or chunk params invalid.
        RuntimeError: If chunking fails.
    """
    if not docs:
        raise ValueError("No documents provided for chunking")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Invalid chunk_size/chunk_overlap values")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        chunks = splitter.split_documents(docs)
    except Exception as e:
        raise RuntimeError("Chunking failed") from e

    doc_chunk_counters: Dict[str, int] = {}
    for c in chunks:
        meta = dict(c.metadata or {})
        did = str(meta.get("doc_id", "unknown"))
        doc_chunk_counters.setdefault(did, 0)
        doc_chunk_counters[did] += 1

        meta["chunk_index"] = doc_chunk_counters[did]
        meta["ingest_run_id"] = ingest_run_id
        meta["ingested_at"] = _utc_now_iso()
        c.metadata = meta

    return chunks


def build_faiss_index(
    chunks: List[Document],
    vectorstore_path: str,
    embedding_model: str,
) -> None:
    """
    Build and persist FAISS index.

    Args:
        chunks: Chunked documents.
        vectorstore_path: Output directory for FAISS index.
        embedding_model: OpenAI embedding model name.

    Raises:
        ValueError: If chunks is empty.
        RuntimeError: If FAISS creation or save fails.
    """
    if not chunks:
        raise ValueError("No chunks provided for indexing")

    try:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    except Exception as e:
        raise RuntimeError("Failed to create FAISS index") from e

    try:
        os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
        vs.save_local(vectorstore_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index to: {vectorstore_path}") from e


def ingest(
    input_path: str,
    vectorstore_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> Tuple[int, int, List[str]]:
    """
    Ingest PDFs into FAISS.

    Args:
        input_path: PDF file or directory containing PDFs.
        vectorstore_path: Output folder to persist FAISS.
        chunk_size: Chunk size.
        chunk_overlap: Chunk overlap.
        embedding_model: Embedding model for vectorization.

    Returns:
        Tuple: (num_docs, num_chunks, doc_ids)

    Raises:
        Exception: Propagates errors from lower-level steps.
    """
    pdf_files = _list_pdf_files(input_path)
    ingest_run_id = str(uuid.uuid4())

    all_pages: List[Document] = []
    doc_ids: List[str] = []

    for pdf in pdf_files:
        doc_id = _compute_doc_id(pdf)
        doc_ids.append(doc_id)
        pages = load_pdf_pages(pdf, doc_id=doc_id)
        all_pages.extend(pages)

    chunks = chunk_documents(
        docs=all_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ingest_run_id=ingest_run_id,
    )

    build_faiss_index(
        chunks=chunks,
        vectorstore_path=vectorstore_path,
        embedding_model=embedding_model,
    )

    return len(doc_ids), len(chunks), doc_ids


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace
    """
    s = load_settings()
    parser = argparse.ArgumentParser(
        prog="ingest_faiss",
        description="Ingest one or many PDFs into a FAISS vector index",
    )
    parser.add_argument(
        "--input",
        default=s.data_dir,
        help='PDF file path or directory path containing PDFs. Default is "data"',
    )
    parser.add_argument(
        "--out",
        default=s.vectorstore_path,
        help='Output directory for FAISS index. Default is "vectorstore/faiss_index"',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=s.chunk_size,
        help="Chunk size in characters. Default from env CHUNK_SIZE",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=s.chunk_overlap,
        help="Chunk overlap in characters. Default from env CHUNK_OVERLAP",
    )
    parser.add_argument(
        "--embedding-model",
        default=s.openai_embedding_model,
        help="Embedding model name. Default from env OPENAI_EMBEDDING_MODEL",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for ingestion.

    Returns:
        Exit code integer.
    """
    load_dotenv()
    settings = load_settings()
    settings.validate_for_openai()

    args = _parse_args(argv)

    try:
        num_docs, num_chunks, doc_ids = ingest(
            input_path=args.input,
            vectorstore_path=args.out,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
        )
        print("Ingestion completed successfully")
        print(f'Input: "{args.input}"')
        print(f'FAISS index saved at: "{args.out}"')
        print(f"Documents ingested: {num_docs}")
        print(f"Chunks indexed: {num_chunks}")
        print("doc_ids:")
        for d in doc_ids[:10]:
            print(f'  - "{d}"')
        if len(doc_ids) > 10:
            print(f"  ... and {len(doc_ids) - 10} more")
        return 0
    except Exception as e:
        print("Ingestion failed")
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))