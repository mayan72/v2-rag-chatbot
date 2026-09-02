"""
Knowledge Base Service

Responsibilities
----------------
1. Validate uploaded files.
2. Extract content from PDF/JPG using Docling.
3. Extract content from CSV.
4. Extract content from XLSX.
5. Split documents into chunks.
6. Create embeddings.
7. Store embeddings in the existing Chroma collection.
8. Replace previously indexed documents.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import pandas as pd

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from rag.table_store import TableStore
from debug_trace import dbg
from logger.console import qlog
from services.vector_cleanup import delete_all_vector_ids

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".csv",
    ".xlsx",
    ".jpg",
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# Service
# ============================================================

class KnowledgeService:

    def __init__(self):

        logger.info(
            "Initializing KnowledgeService..."
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        self.vector_db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_DB_PATH,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.table_store = TableStore()

        logger.info(
            "KnowledgeService initialized successfully."
        )

    # ========================================================
    # Validation
    # ========================================================

    def validate_file(
        self,
        filename: str,
        file_size: int,
    ):

        extension = Path(
            filename
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:

            raise ValueError(
                "Unsupported file type. "
                "Allowed: PDF, CSV, XLSX and JPG."
            )

        if file_size > MAX_FILE_SIZE:

            raise ValueError(
                "File size exceeds the 10 MB limit."
            )

    # ========================================================
    # Document ID
    # ========================================================

    def _create_document_id(
        self,
        filename: str,
    ) -> str:

        filename = filename.lower()

        filename = re.sub(
            r"[^a-z0-9.]+",
            "_",
            filename,
        ).strip("_")

        return f"uploaded_{filename.replace('.', '_')}"

    # ========================================================
    # CSV
    # ========================================================

    def _load_csv(
        self,
        file_path: Path,
        document_id: str,
        filename: str,
    ) -> List[Document]:

        logger.info(
            "Reading CSV: %s",
            filename,
        )

        df = pd.read_csv(file_path)

        documents = []

        for index, row in df.iterrows():

            values = []

            for column in df.columns:

                value = row.get(column)

                if pd.isna(value):
                    continue

                values.append(
                    f"{column}: {value}"
                )

            text = "\n".join(values).strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "csv",
                        "document_name": filename,
                        "document_id": document_id,
                        "row_number": index + 1,
                    },
                )
            )

        return documents

    # ========================================================
    # XLSX
    # ========================================================

    def _load_xlsx(
        self,
        file_path: Path,
        document_id: str,
        filename: str,
    ) -> List[Document]:

        logger.info(
            "Reading XLSX: %s",
            filename,
        )

        workbook = pd.ExcelFile(
            file_path
        )

        documents = []

        for sheet_name in workbook.sheet_names:

            logger.info(
                "Processing sheet: %s",
                sheet_name,
            )

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
            )

            for index, row in df.iterrows():

                values = []

                for column in df.columns:

                    value = row.get(column)

                    if pd.isna(value):
                        continue

                    values.append(
                        f"{column}: {value}"
                    )

                text = "\n".join(
                    values
                ).strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_type": "xlsx",
                            "document_name": filename,
                            "document_id": document_id,
                            "sheet_name": str(
                                sheet_name
                            ),
                            "row_number": index + 1,
                        },
                    )
                )

        return documents

    # ========================================================
    # Docling
    # ========================================================

    def _load_docling_document(
        self,
        file_path: Path,
        document_id: str,
        filename: str,
        source_type: str,
    ) -> List[Document]:

        logger.info(
            "Processing with Docling: %s",
            filename,
        )

        converter = DocumentConverter()

        result = converter.convert(
            str(file_path)
        )

        markdown = (
            result.document
            .export_to_markdown()
        ).strip()

        if not markdown:
            return []

        markdown = self._flatten_markdown_tables(markdown)

        return [
            Document(
                page_content=markdown,
                metadata={
                    "source_type": source_type,
                    "document_name": filename,
                    "document_id": document_id,
                },
            )
        ]

    def _looks_like_amount(self, text: str) -> bool:
        text = text.strip().replace(",", "")
        if not text:
            return False
        if text[:1] in "+-":
            text = text[1:]
        if text.startswith("$"):
            text = text[1:]
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _collapse_table_row(self, line: str) -> str:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell and set(cell) != {"-"}]
        unique = []
        for cell in cells:
            if not unique or unique[-1] != cell:
                unique.append(cell)
        if len(unique) >= 2 and self._looks_like_amount(unique[-1]):
            return f"{unique[-2]}: {unique[-1]}"
        return " | ".join(unique)

    def _flatten_markdown_tables(self, markdown: str) -> str:
        lines = markdown.splitlines()
        out = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if "|" in line:
                while index < len(lines) and "|" in lines[index]:
                    collapsed = self._collapse_table_row(lines[index])
                    if collapsed:
                        out.append(collapsed)
                    index += 1
                continue
            out.append(line)
            index += 1
        return "\n".join(out)

    # ========================================================
    # Extract
    # ========================================================

    def _extract_documents(
        self,
        file_path: Path,
        filename: str,
        document_id: str,
    ) -> List[Document]:

        extension = Path(filename).suffix.lower() or file_path.suffix.lower()

        if extension == ".csv":

            return self._load_csv(
                file_path,
                document_id,
                filename,
            )

        if extension == ".xlsx" or extension == ".xls":

            return self._load_xlsx(
                file_path,
                document_id,
                filename,
            )

        if extension == ".pdf":

            return self._load_docling_document(
                file_path,
                document_id,
                filename,
                "pdf",
            )

        if extension == ".jpg":

            return self._load_docling_document(
                file_path,
                document_id,
                filename,
                "jpg",
            )

        raise ValueError(
            f"Unsupported extension: {extension}"
        )

    # ========================================================
    # Delete previous document
    # ========================================================

    def _remove_existing_document(
        self,
        document_id: str,
    ):

        logger.info(
            "Checking existing vectors for document_id=%s",
            document_id,
        )

        result = self.vector_db.get(
            where={
                "document_id": document_id
            }
        )

        ids = result.get(
            "ids",
            [],
        )

        if ids:

            logger.info(
                "Removing %d existing chunks.",
                len(ids),
            )

            self.vector_db.delete(
                ids=ids
            )

    def clear_all(self) -> dict:
        """
        Remove stored embeddings and uploaded table files.

        Does not change the upload/index path. New files are stored
        with the same collection and table_store as before.
        """

        deleted_chunks = delete_all_vector_ids(self.vector_db)

        persist = getattr(self.vector_db, "persist", None)
        if callable(persist):
            try:
                persist()
            except Exception:
                logger.exception("Unable to persist Chroma after clear.")

        deleted_tables = self.table_store.clear_all()

        qlog(
            "CLEAR",
            chunks=deleted_chunks,
            tables=deleted_tables,
        )

        return {
            "deleted_chunks": deleted_chunks,
            "deleted_tables": deleted_tables,
            "status": "cleared",
        }

    # ========================================================
    # Index
    # ========================================================

    def index_document(
        self,
        file_path: Path,
        filename: str,
    ) -> dict:

        document_id = (
            self._create_document_id(
                filename
            )
        )

        logger.info(
            "Indexing document: %s",
            filename,
        )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        documents = self._extract_documents(
            file_path,
            filename,
            document_id,
        )

        if not documents:

            raise ValueError(
                "No readable content found in the document."
            )

        logger.info(
            "Extracted %d documents.",
            len(documents),
        )

        extension = Path(filename).suffix.lower() or file_path.suffix.lower()

        if extension in {".csv", ".xlsx", ".xls"}:
            self.table_store.delete(document_id)
            saved = self.table_store.upsert_from_file(
                file_path=file_path,
                filename=filename,
                document_id=document_id,
            )
            if not saved:
                raise ValueError(
                    "Spreadsheet rows were extracted for search, "
                    "but no table was written to table_store. "
                    "Check that openpyxl is installed and the sheet is not empty."
                )
            table_dir = self.table_store.root / document_id
            chunks = documents
            logger.info(
                "Table store saved | path=%s | rows=%s",
                table_dir,
                saved.get("row_count"),
            )
            dbg(
                "INDEX_TABLE_SAVED",
                filename=filename,
                document_id=document_id,
                extension=extension,
                table_dir=str(table_dir),
                row_count=saved.get("row_count"),
                table_chunks=len(chunks),
            )
        else:
            chunks = self.splitter.split_documents(
                documents
            )
            dbg(
                "INDEX_DOC_CHUNKED",
                filename=filename,
                document_id=document_id,
                extension=extension,
                chunks=len(chunks),
            )

        logger.info(
            "Created %d chunks.",
            len(chunks),
        )

        if not chunks:

            raise ValueError(
                "No chunks were created from the document."
            )

        # ----------------------------------------------------
        # Replace existing document
        # ----------------------------------------------------

        self._remove_existing_document(
            document_id
        )

        # ----------------------------------------------------
        # Add to existing Chroma collection
        # ----------------------------------------------------

        logger.info(
            "Creating embeddings and storing vectors..."
        )

        self.vector_db.add_documents(
            documents=chunks
        )

        qlog(
            "INDEX",
            document=filename,
            chunks=len(chunks),
            type=extension,
        )

        result = {
            "document_id": document_id,
            "document_name": filename,
            "chunks_created": len(chunks),
            "status": "completed",
        }
        if extension in {".csv", ".xlsx", ".xls"}:
            result["table_store_path"] = str(
                self.table_store.root / document_id / "data.jsonl"
            )
        return result