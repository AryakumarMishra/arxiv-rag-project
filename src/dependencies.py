from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.db.factory import make_database
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.client import CacheClient
from src.services.cache.factory import make_cache_client
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.embeddings.factory import make_embeddings_client
from src.services.langfuse.client import LangfuseTracer
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.client import OllamaClient
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.parser import PDFParserService
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.indexing.hybrid_indexer import HybridIndexerService
from src.services.indexing.factory import make_hybrid_indexer_service

# --- Service Caching (Singleton Pattern) ---
_db_cache: BaseDatabase | None = None
_opensearch_client_cache: OpenSearchClient | None = None
_arxiv_client_cache: ArxivClient | None = None
_pdf_parser_service_cache: PDFParserService | None = None
_embeddings_service_cache: JinaEmbeddingsClient | None = None
_ollama_client_cache: OllamaClient | None = None
_langfuse_tracer_cache: LangfuseTracer | None = None
_cache_client_cache: CacheClient | None = None
_indexer_service_cache: HybridIndexerService | None = None

# --- Dependency Getters ---

def get_database() -> BaseDatabase:
    global _db_cache
    if _db_cache is None:
        _db_cache = make_database()
    return _db_cache

def get_opensearch_client() -> OpenSearchClient:
    global _opensearch_client_cache
    if _opensearch_client_cache is None:
        _opensearch_client_cache = make_opensearch_client()
    return _opensearch_client_cache

def get_arxiv_client() -> ArxivClient:
    global _arxiv_client_cache
    if _arxiv_client_cache is None:
        _arxiv_client_cache = make_arxiv_client()
    return _arxiv_client_cache

def get_pdf_parser_service() -> PDFParserService:
    global _pdf_parser_service_cache
    if _pdf_parser_service_cache is None:
        _pdf_parser_service_cache = make_pdf_parser_service()
    return _pdf_parser_service_cache

def get_embeddings_service() -> JinaEmbeddingsClient:
    global _embeddings_service_cache
    if _embeddings_service_cache is None:
        _embeddings_service_cache = make_embeddings_client()
    return _embeddings_service_cache

def get_ollama_client() -> OllamaClient:
    global _ollama_client_cache
    if _ollama_client_cache is None:
        _ollama_client_cache = make_ollama_client()
    return _ollama_client_cache

def get_langfuse_tracer() -> LangfuseTracer:
    global _langfuse_tracer_cache
    if _langfuse_tracer_cache is None:
        _langfuse_tracer_cache = make_langfuse_tracer()
    return _langfuse_tracer_cache

def get_cache_client() -> CacheClient | None:
    global _cache_client_cache
    if _cache_client_cache is None:
        # Provide Settings to the cache client factory
        _cache_client_cache = make_cache_client(get_settings())
    return _cache_client_cache

# --- Annotated Dependency Types ---
# Define all dependency types BEFORE they are used in function signatures.
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser_service)]

# --- Functions that use Annotated Types ---

def get_db_session(database: DatabaseDep = Depends()) -> Session:
    with database.get_session() as session:
        yield session

def get_indexer_service() -> HybridIndexerService:
    """Get a cached instance of the hybrid indexer service."""
    global _indexer_service_cache
    if _indexer_service_cache is None:
        # Factory constructs its own dependencies from Settings
        _indexer_service_cache = make_hybrid_indexer_service()
    return _indexer_service_cache

# --- Final Annotated Types ---
SessionDep = Annotated[Session, Depends(get_db_session)]
IndexerDep = Annotated[HybridIndexerService, Depends(get_indexer_service)]