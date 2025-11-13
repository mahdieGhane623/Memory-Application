"""
Agentic Memory Service with Azure AI Search

"""

import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
import json
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os

# Azure imports
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration,
    SemanticConfiguration, SemanticField, SemanticPrioritizedFields, SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, status, Body, Query
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

# Local imports
from llm_controller import LLMController
from models.auth import Token
from auth import authenticate_user, create_access_token, get_current_user
from config import (
    AZURE_BLOB_CONN_STR, AZURE_BLOB_CONTAINER,
    MAX_DIRECT_PAYLOAD_BYTES, OPENAI_API_KEY, AUTH_USERNAME,
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(title="LongTermMemoryService with Azure AI Search")

# --- Pydantic Models ---
class MemoryPayload(BaseModel):
    session_key: str
    user_id: str
    memory_type: str
    payload: Dict[str, Any]
    request_id: str

class SearchQuery(BaseModel):
    query: str
    k: int = 5
    include_neighbors: bool = False

class MemoryNote:
    """A memory note that represents a single unit of information."""
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 session_key: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[List[str]] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 memory_version: Optional[int] = None,
                 audit: Optional[List[Dict]] = None,
                 **kwargs):
        self.session_key = session_key
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        current_time = datetime.utcnow().isoformat() + "Z"
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []
        self.memory_version = memory_version or 1
        self.audit = audit or []
        
        # Store any additional kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "session_key": self.session_key,
            "content": self.content,
            "keywords": self.keywords,
            "links": self.links,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "evolution_history": self.evolution_history,
            "category": self.category,
            "tags": self.tags,
            "memory_version": self.memory_version,
            "audit": self.audit
        }

class AzureAISearchRetriever:
    """Azure AI Search retriever with vector and hybrid search capabilities."""
    
    def __init__(
        self,
        search_endpoint: str,
        search_key: str,
        index_name: str = "memory-index",
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """Initialize Azure AI Search retriever."""
        self.model = SentenceTransformer(model_name)
        self.index_name = index_name
        self.embedding_dimensions = self.model.get_sentence_embedding_dimension()
        
        credential = AzureKeyCredential(search_key)
        
        # Client for index management
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=credential
        )
        
        # Client for document operations
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential
        )
        
        # Create or update index
        self._ensure_index()
        
        logger.info(f"Azure AI Search retriever initialized with index: {index_name}")
    
    def _ensure_index(self):
        """Create or update the search index with vector search configuration."""
        
        # Vector search configuration
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="memory-vector-profile",
                    algorithm_configuration_name="memory-hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="memory-hnsw-config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )
        
        # Semantic search configuration
        semantic_config = SemanticConfiguration(
            name="memory-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="category"),
                content_fields=[
                    SemanticField(field_name="content"),
                    SemanticField(field_name="context")
                ],
                keywords_fields=[
                    SemanticField(field_name="keywords"),
                    SemanticField(field_name="tags")
                ]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Index schema
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True, analyzer_name="standard.lucene"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dimensions,
                vector_search_profile_name="memory-vector-profile"
            ),
            SearchField(name="session_key", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(name="user_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(name="memory_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(name="request_id", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="keywords", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=True),
            SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=True, facetable=True),
            SearchField(name="links", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
            SearchField(name="context", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="category", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True),
            SearchField(name="timestamp", type=SearchFieldDataType.String, sortable=True, filterable=True),
            SearchField(name="last_accessed", type=SearchFieldDataType.String, sortable=True),
            SearchField(name="retrieval_count", type=SearchFieldDataType.Int32, sortable=True, filterable=True),
            SearchField(name="memory_version", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(name="blob_url", type=SearchFieldDataType.String, filterable=False),
            SearchField(name="size_bytes", type=SearchFieldDataType.Int64, filterable=True),
            SearchField(name="evolution_history", type=SearchFieldDataType.String, filterable=False),
            SearchField(name="audit", type=SearchFieldDataType.String, filterable=False),
        
        ]
        
        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            self.index_client.create_or_update_index(index)
            logger.info(f"Index '{self.index_name}' created/updated successfully")
        except Exception as e:
            logger.error(f"Failed to create/update index: {e}")
            raise
    
    def add_document(self, content: str, metadata: Dict, doc_id: str):
        """Add a document with its embedding and metadata."""
        try:
            embedding = self.model.encode(content).tolist()

            processed_metadata = {}
            for key, value in metadata.items():
                if key == 'id':
                    continue
                if key in ['evolution_history', 'audit']:
                    # Serialize lists/dicts to JSON string
                    #processed_metadata[key] = json.dumps(value) if value else "[]"
                    if value is None:
                        processed_metadata[key] = "[]"
                    elif isinstance(value, (list, dict)):
                        processed_metadata[key] = json.dumps(value)
                    else:
                        processed_metadata[key] = str(value)
                elif key in ['keywords', 'tags', 'links']:
                    # Ensure lists are proper lists (not None)
                    #processed_metadata[key] = value if isinstance(value, list) else []
                    if value is None:
                        processed_metadata[key] = []
                    elif isinstance(value, list):
                        processed_metadata[key] = value
                    else:
                        processed_metadata[key] = [str(value)]
                else:
                    processed_metadata[key] = value
            
            document = {
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                **processed_metadata
            }
            debug_doc = {k: v for k, v in document.items() if k != 'embedding'}
            logger.debug(f"Uploading document {doc_id}")
            logger.debug(f"Fields: {list(debug_doc.keys())}")
            logger.debug(f"Sample data: keywords={debug_doc.get('keywords')}, tags={debug_doc.get('tags')}")
            
            result = self.search_client.upload_documents(documents=[document])
            
            if result[0].succeeded:
                logger.debug(f"Document {doc_id} added successfully")
            else:
                logger.error(f"Failed to add document {doc_id}: {result[0].error_message}")
                
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            raise
    
    def search(self, query: str, k: int = 5, search_mode: str = "hybrid") -> Dict[str, Any]:
        """Search documents using vector, text, or hybrid search."""
        try:
            # Handle special query formats
            filter_expr = None
            search_text = query
            
            if ":" in query:
                key, val = query.split(":", 1)
                key = key.strip()
                val = val.strip()
                
                if key in ("session_key", "request_id", "user_id", "memory_type", "category"):
                    filter_expr = f"{key} eq '{val}'"
                    search_text = "*"
                    search_mode = "text"
            
            results_list = []
            
            if search_mode in ("vector", "hybrid"):
                query_embedding = self.model.encode(query).tolist()
                
                vector_results = self.search_client.search(
                    search_text=None if search_mode == "vector" else search_text,
                    vector_queries=[{
                        "kind": "vector",
                        "vector": query_embedding,
                        "k_nearest_neighbors": k,
                        "fields": "embedding"
                    }],
                    filter=filter_expr,
                    top=k,
                    select=["id", "content", "session_key", "user_id", "memory_type",
                           "request_id", "keywords", "tags", "links", "context",
                           "category", "timestamp", "last_accessed", "retrieval_count",
                           "memory_version", "blob_url", "size_bytes"]
                )
                results_list = list(vector_results)
            
            elif search_mode == "text":
                text_results = self.search_client.search(
                    search_text=search_text,
                    filter=filter_expr,
                    top=k,
                    select=["id", "content", "session_key", "user_id", "memory_type",
                           "request_id", "keywords", "tags", "links", "context",
                           "category", "timestamp", "last_accessed", "retrieval_count",
                           "memory_version", "blob_url", "size_bytes"]
                )
                results_list = list(text_results)
            
            if not results_list:
                return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
            
            ids = []
            distances = []
            metadatas = []
            
            for result in results_list:
                ids.append(result["id"])
                score = result.get("@search.score", 0.0)
                #distance = 1.0 - (score / 100.0) if score > 0 else 1.0
                if score > 0:
                    distance = 1.0 - (score / 1.0)  # Normalize and invert
                else:
                    distance = 1.0
                distances.append(distance)
                
                metadata = {k: v for k, v in result.items() if k not in ["@search.score", "@search.reranker_score"]}
                metadatas.append(metadata)
            
            return {"ids": [ids], "distances": [distances], "metadatas": [metadatas]}
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID."""
        try:
            result = self.search_client.get_document(key=doc_id)
            return result
        except Exception as e:
            logger.warning(f"Document {doc_id} not found: {e}")
            return None
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]):
        """Update a document."""
        try:
            # Get existing document
            existing = self.get_document(doc_id)
            if not existing:
                raise ValueError(f"Document {doc_id} not found")
            
            # Process updates - serialize complex fields
            processed_updates = {}
            for key, value in updates.items():
                if key == 'id':
                    continue

                if key in ['evolution_history', 'audit']:
                #    processed_updates[key] = json.dumps(value) if value else "[]"
                    if value is None:
                        processed_updates[key] = "[]"
                    elif isinstance(value, (list, dict)):
                        processed_updates[key] = json.dumps(value)
                    else:
                        processed_updates[key] = str(value)

                elif key in ['keywords', 'tags', 'links']:
                #    processed_updates[key] = value if isinstance(value, list) else []
                    if value is None:
                        processed_updates[key] = []
                    elif isinstance(value, list):
                        processed_updates[key] = value
                    else:
                        # If it's a string or other type, wrap in list
                        processed_updates[key] = [str(value)]
                
                else:
                    processed_updates[key] = value
            
            # Merge updates
            processed_existing = {}
            for key, value in existing.items():
                if key == 'id':
                    continue
                if key in ['evolution_history', 'audit']:
                    # If it's already a string, keep it; if it's a dict/list, serialize it
                    if isinstance(value, str):
                        processed_existing[key] = value
                    elif isinstance(value, (list, dict)):
                        processed_existing[key] = json.dumps(value)
                    else:
                        processed_existing[key] = str(value) if value is not None else "[]"
                elif key in ['keywords', 'tags', 'links']:
                    processed_existing[key] = value if isinstance(value, list) else []
                else:
                    processed_existing[key] = value
            
            # Merge updates
            updated_doc = {**processed_existing, **processed_updates, "id": doc_id}
            
            #updated_doc = {**existing, **updates, "id": doc_id}
            
            # Re-generate embedding if content changed
            if "content" in updates:
                updated_doc["embedding"] = self.model.encode(updates["content"]).tolist()
            
            logger.debug(f"Updating document {doc_id} with fields: {list(processed_updates.keys())}")
           
            # Upload updated document
            result = self.search_client.upload_documents(documents=[updated_doc])
            
            if not result[0].succeeded:
                raise RuntimeError(f"Failed to update: {result[0].error_message}")
            else:
                logger.info(f"✓ Document {doc_id} updated successfully")
               
                
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise
    
    def delete_document(self, doc_id: str):
        """Delete a document by ID."""
        try:
            self.search_client.delete_documents(documents=[{"id": doc_id}])
            logger.debug(f"Document {doc_id} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise

class AgenticMemorySystem:
    """Core memory system with Azure AI Search backend."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 search_endpoint: str = None,
                 search_key: str = None,
                 search_index: str = "memory-index"):
        
        self.model_name = model_name
        self.retriever = AzureAISearchRetriever(
            search_endpoint=search_endpoint,
            search_key=search_key,
            index_name=search_index,
            model_name=model_name
        )
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
        # Blob storage setup
        self.blob_client: Optional[BlobServiceClient] = None
        if AZURE_BLOB_CONN_STR:
            self.blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
            try:
                self.blob_client.create_container(AZURE_BLOB_CONTAINER)
            except Exception:
                pass
        
        self._evolution_system_prompt = '''
You are an AI memory evolution agent. Analyze the new memory and its nearest neighbors.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbors:
{nearest_neighbors_memories}

Determine:
1. Should this memory evolve?
2. What actions: ["strengthen", "update_neighbor"]
3. Which memories to connect?
4. What tags to update?

Return JSON:
{{
    "should_evolve": true/false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["memory_id1", "memory_id2"],
    "tags_to_update": ["tag1", "tag2"],
    "new_context_neighborhood": ["context1", "context2"],
    "new_tags_neighborhood": [["tag1"], ["tag2"]]
}}
'''
    
    def _upload_json_to_blob(self, json_bytes: bytes, session_key: str) -> str:
        """Upload JSON payload to Azure Blob Storage."""
        if not self.blob_client:
            raise RuntimeError("Blob storage not configured")
        blob_name = f"{session_key}/memory_{uuid.uuid4().hex}.json"
        container_client = self.blob_client.get_container_client(AZURE_BLOB_CONTAINER)
        cs = ContentSettings(content_type="application/json; charset=utf-8")
        container_client.upload_blob(name=blob_name, data=json_bytes, overwrite=True, content_settings=cs)
        account_url = self.blob_client.url.rstrip("/")
        return f"{account_url}/{AZURE_BLOB_CONTAINER}/{blob_name}"
    
    def _fetch_payload_from_blob(self, blob_url: str) -> Dict[str, Any]:
        """Fetch JSON payload from Azure Blob Storage."""
        try:
            if not self.blob_client:
                raise RuntimeError("Blob storage not configured")
            
            parts = blob_url.split('/')
            container_name = parts[-2]
            blob_name = '/'.join(parts[-2:]).split('/', 1)[1]
            
            blob_client = self.blob_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob().readall()
            return json.loads(blob_data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error fetching blob {blob_url}: {e}")
            raise RuntimeError(f"Failed to fetch blob: {str(e)}")
    
    def _deep_merge(self, dest: dict, patch: dict) -> dict:
        """Deep merge two dictionaries."""
        for k, v in patch.items():
            if k in dest and isinstance(dest[k], dict) and isinstance(v, dict):
                self._deep_merge(dest[k], v)
            else:
                dest[k] = v
        return dest
    
    def analyze_content(self, content: str) -> Dict:
        """Analyze content using LLM to extract semantic metadata."""
        prompt = """Generate a structured analysis of the following content:
1. Identify salient keywords (nouns, verbs, key concepts)
2. Extract core themes and context
3. Create relevant tags

Format as JSON:
{
    "keywords": ["keyword1", "keyword2"],
    "context": "Main topic and purpose",
    "tags": ["tag1", "tag2"]
}

Content: """ + content
        
        try:
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }}
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}
    
    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        if time is not None:
            kwargs['timestamp'] = time
        
        note = MemoryNote(content=content, **kwargs)
        
        # Analyze content if needed
        needs_analysis = not note.keywords or note.context == "General" or not note.tags
        if needs_analysis:
            try:
                analysis = self.analyze_content(content)
                if not note.keywords:
                    note.keywords = analysis.get("keywords", [])
                if note.context == "General":
                    note.context = analysis.get("context", "General")
                if not note.tags:
                    note.tags = analysis.get("tags", [])
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # Process memory evolution
        evo_label, note = self.process_memory(note)
        
        # Store in Azure AI Search
        metadata = note.to_dict()
        del metadata['content']
        self.retriever.add_document(note.content, metadata, note.id)
        
        if evo_label:
            self.evo_cnt += 1
        
        return note.id
    
    def read(self, memory_id: str) -> Optional[Dict]:
        """Retrieve a memory by ID."""
        return self.retriever.get_document(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note."""
        try:
            self.retriever.update_document(memory_id, kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note."""
        try:
            self.retriever.delete_document(memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories."""
        results = self.retriever.search(query, k)
        
        memories = []
        if isinstance(results, dict) and "metadatas" in results and results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"][0]):
                memory_dict = {k: v for k, v in metadata.items() if k not in ['embedding']}
                if "ids" in results and i < len(results["ids"][0]):
                    memory_dict["id"] = str(results["ids"][0][i])
                if "distances" in results and i < len(results["distances"][0]):
                    #memory_dict["score"] = results["distances"][0][i]
                    distance = results["distances"][0][i]
                    memory_dict["score"] = distance  # This is actually distance
                    memory_dict["similarity"] = 1.0 - distance  # Add actual similarity too
                    
                memories.append(memory_dict)
        
        return memories[:k]
    
    def search_with_neighbors(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search that includes linked neighbor memories."""
        primary_results = self.search(query, k)
        
        seen_ids = {r["id"] for r in primary_results}
        all_results = primary_results.copy()
        
        for result in primary_results:
            if len(all_results) >= k * 2:
                break
            
            links = result.get("links", [])
            for link_id in links:
                if link_id not in seen_ids:
                    neighbor = self.read(link_id)
                    if neighbor:
                        neighbor_dict = {k: v for k, v in neighbor.items() if k not in ['embedding']}
                        neighbor_dict["is_neighbor"] = True
                        neighbor_dict["linked_from"] = result["id"]
                        all_results.append(neighbor_dict)
                        seen_ids.add(link_id)
        
        return all_results
    
    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process memory evolution."""
        try:
            # Search for similar memories
            similar = self.search(note.content, k=5)
            if not similar:
                logger.debug(f"No similar memories found for {note.id}")
                return False, note
            logger.info(f"Processing memory {note.id}, found {len(similar)} similar memories")
    
            auto_link_threshold = 0.3  # Link if similarity > 70%
            #for mem in similar:
            #    similarity_score = 1.0 - mem.get('score', 1.0)  # Convert distance to similarity
            #    if similarity_score > auto_link_threshold and mem.get('id') != note.id:
            #        if mem['id'] not in note.links:
            #            note.links.append(mem['id'])
            #            logger.debug(f"Auto-linked {note.id} -> {mem['id']} (similarity: {similarity_score:.2f})")
            for i, mem in enumerate(similar):
                # Azure Search returns 'score' which is actually a distance (lower = more similar)
                # Values near 0 = very similar, values near 1 = dissimilar
                distance = 1.0 - mem.get('score', 1.0)
                similarity_score = 1.0 - distance  # Convert to similarity (higher = more similar)
                
                mem_id = mem.get('id')
                
                # Don't link to self
                if mem_id == note.id:
                    logger.debug(f"Skipping self-link for {note.id}")
                    continue
                
                # Check distance threshold (not similarity)
                if distance < auto_link_threshold:
                    if mem_id not in note.links:
                        note.links.append(mem_id)
                        logger.info(f"✓ Auto-linked {note.id} -> {mem_id} (distance: {distance:.4f}, similarity: {similarity_score:.2%})")
                else:
                    logger.debug(f"Memory {mem_id} too dissimilar (distance: {distance:.4f})")
            
            logger.info(f"Memory {note.id} now has {len(note.links)} auto-links")
            
            # LLM-BASED EVOLUTION: Ask LLM for deeper analysis (optional enhancement)
            if len(similar) > 0:
                neighbors_text = "\n".join([
                    f"Memory {i} (ID: {m.get('id', 'unknown')}): {m.get('content', '')[:200]}... Context: {m.get('context', '')}"
                    for i, m in enumerate(similar)
                ])
            # LLM-BASED EVOLUTION: Ask LLM for deeper analysis
            #neighbors_text = "\n".join([
            #    f"Memory {i}: {m.get('content', '')[:200]}... Context: {m.get('context', '')}"
            #    for i, m in enumerate(similar)
            #])
            
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(similar)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "string"}},
                                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {"type": "array"}
                            }
                        }
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json.get("should_evolve", False)
                
                if should_evolve:
                    actions = response_json.get("actions", [])
                    if "strengthen" in actions:
                        # Add LLM-suggested links (in addition to auto-links)
                        llm_links = response_json.get("suggested_connections", [])
                        for link_id in llm_links:
                            if link_id not in note.links:
                                note.links.append(link_id)
                                logger.info(f"LLM-linked {note.id} -> {link_id}")
                        
                        # Update tags if suggested
                        new_tags = response_json.get("tags_to_update", [])
                        if new_tags:
                            note.tags = new_tags
                            logger.info(f"Updated tags for {note.id}: {new_tags}")
                    
                
                return should_evolve or len(note.links) > 0, note
                
            except Exception as e:
                logger.warning(f"LLM evolution failed, using auto-links only: {e}")
                # Return with auto-links even if LLM fails
                return len(note.links) > 0, note
                
        except Exception as e:
            logger.error(f"Error in process_memory for {note.id}: {e}")
            return False, note
        
    def write_longterm_memory(self, session_key: str, user_id: str, memory_type: str, 
                            payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Save a memory note with idempotency check."""
        # Check if already exists
        search_results = self.search(f"request_id:{request_id}", k=1)
        if search_results and any(request_id in result.get('tags', []) for result in search_results):
            result = search_results[0]
            return {
                "status": "ok",
                "message": "already_exists",
                "id": result['id'],
                "created_at": result.get('timestamp'),
                "request_id": request_id
            }
        
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        metadata = {
            "session_key": session_key,
            "user_id": user_id,
            "memory_type": memory_type,
            "request_id": request_id,
            "memory_version": 1
        }
        
        # Handle large payloads
        if len(payload_bytes) > MAX_DIRECT_PAYLOAD_BYTES:
            try:
                blob_url = self._upload_json_to_blob(payload_bytes, session_key)
                content = f"Payload stored in blob: {blob_url}"
                metadata.update({"blob_url": blob_url, "size_bytes": len(payload_bytes)})
            except Exception as e:
                raise RuntimeError(f"Blob upload failed: {str(e)}")
        else:
            content = json.dumps(payload)
        
        note_id = self.add_note(
            content=content,
            category=memory_type,
            context=f"Session: {session_key}, User: {user_id}",
            tags=[session_key, user_id, memory_type, request_id],
            timestamp=datetime.utcnow().isoformat() + "Z",
            **metadata
        )
        
        note = self.read(note_id)
        return {
            "status": "ok",
            "id": note_id,
            "created_at": note.get("timestamp") if note else datetime.utcnow().isoformat() + "Z",
            "request_id": request_id
        }
    
    def get_session(self, session_key: str, k: int = 100) -> Dict[str, Any]:
        """Retrieve the most recent memory for a session."""
        results = self.search(f"session_key:{session_key}", k=k)
        
        if not results:
            raise ValueError("Session not found")
        
        # Get most recent by timestamp
        doc = max(results, key=lambda x: x.get('timestamp', ''))
        
        payload = doc.get("content")
        if doc.get("blob_url"):
            doc["payload_pointer"] = doc.get("blob_url")
        
        tags = doc.get("tags", []) or []
        user_id_tag = next((t for t in tags if t != session_key and t != doc.get("category") 
                           and t != doc.get("request_id", "")), "unknown")
        request_id_tag = next((t for t in tags if t not in [session_key, doc.get("category", ""), 
                              user_id_tag]), "unknown")
        
        return {
            "status": "ok",
            "doc": {
                "id": doc.get("id"),
                "session_key": session_key,
                "user_id": user_id_tag,
                "memory_type": doc.get("category", "unknown"),
                "payload": payload,
                "created_at": doc.get("timestamp"),
                "request_id": request_id_tag,
                "memory_version": doc.get("memory_version", 1)
            }
        }
    
    def patch_memory(self, session_key: str, user_id: Optional[str], patch: Dict[str, Any],
                    request_id: Optional[str] = None, expected_version: Optional[int] = None) -> Dict[str, Any]:
        """Update a memory with deep-merge patching and versioning."""
        results = self.search(f"session_key:{session_key}", k=1)
        
        # If not found, create new
        if not results:
            doc_id = self.add_note(
                content=json.dumps(patch),
                category="session_patch_init",
                context=f"Session: {session_key}, User: {user_id or 'unknown'}",
                tags=[session_key, user_id or "unknown", "session_patch_init", 
                      request_id or f"patch_{uuid.uuid4().hex[:8]}"],
                timestamp=datetime.utcnow().isoformat() + "Z",
                memory_version=1,
                session_key=session_key,
                user_id=user_id or "unknown",
                memory_type="session_patch_init",
                request_id=request_id or f"patch_{uuid.uuid4().hex[:8]}"
            )
            return {"status": "ok", "message": "created", "id": doc_id, "memory_version": 1}
        
        base = results[0]
        memory_id = base.get("id")
        current_version = base.get("memory_version", 1)
        
        # Version check
        if expected_version is not None and expected_version != current_version:
            raise ValueError(f"Version mismatch (expected {expected_version}, got {current_version})")
        
        # Get current payload (handle blob URLs)
        if base.get("blob_url"):
            try:
                base_payload = self._fetch_payload_from_blob(base["blob_url"])
            except Exception as e:
                logger.warning(f"Could not fetch blob, starting with patch: {e}")
                base_payload = {}
        else:
            try:
                content = base.get("content", "{}")
                if content.startswith("Payload stored in blob:"):
                    blob_url = content.split("Payload stored in blob: ")[1].strip()
                    base_payload = self._fetch_payload_from_blob(blob_url)
                else:
                    base_payload = json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not parse content, starting with patch: {e}")
                base_payload = {}
        
        # Merge payloads
        merged = self._deep_merge(base_payload.copy(), patch) if base_payload else patch
        
        # Prepare new version
        payload_bytes = json.dumps(merged, ensure_ascii=False).encode("utf-8")
        new_version = current_version + 1
        
        update_data = {
            "memory_version": new_version,
            "last_accessed": datetime.utcnow().isoformat() + "Z"
        }
        
        # Handle large payloads
        if len(payload_bytes) > MAX_DIRECT_PAYLOAD_BYTES:
            try:
                blob_url = self._upload_json_to_blob(payload_bytes, session_key)
                update_data["content"] = f"Payload stored in blob: {blob_url}"
                update_data["blob_url"] = blob_url
                update_data["size_bytes"] = len(payload_bytes)
            except Exception as e:
                raise RuntimeError(f"Blob upload failed: {str(e)}")
        else:
            update_data["content"] = json.dumps(merged)
            update_data["blob_url"] = None
        
        # Add audit trail
        current_audit = base.get("audit", [])
        if not isinstance(current_audit, list):
            current_audit = []
        
        audit_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "op": "patch",
            "request_id": request_id or "",
            "patch_keys": list(patch.keys())
        }
        update_data["audit"] = current_audit + [audit_entry]
        
        # Perform update
        success = self.update(memory_id, **update_data)
        
        if not success:
            raise RuntimeError("Failed to update memory in database")
        
        return {
            "status": "ok",
            "message": "patched",
            "id": memory_id,
            "memory_version": new_version
        }
    
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory note by ID."""
        success = self.delete(memory_id)
        if not success:
            raise ValueError("Memory not found")
        return {"status": "ok", "message": "deleted", "id": memory_id}
##Added two classes for optimize onboard to match and match to onboard requests
class OptimizeOnboardToMatchRequest(BaseModel):
    query: str                                 # user question or short context
    session_filter: Optional[str] = None       # optional session_key filter (if you have it)
    selectors: Optional[List[str]] = None      # list of metadata keys to keep verbatim (e.g. ["customer","intent","id"])
    top_k: int = 8                             # how many candidate memories to retrieve
    summarize: bool = True
    summary_max_tokens: int = 120
    # If you want the matching agent to only see relevant grants:
    grant_only: bool = True
    grant_relevance_threshold: float = 0.6     # 0-1 (higher = stricter). Uses normalized score/distance heuristic.

class OptimizeMatchToOnboardRequest(BaseModel):
    query: str
    session_filter: Optional[str] = None
    selectors: Optional[List[str]] = None
    top_k: int = 8
    summarize: bool = True
    summary_max_tokens: int = 120
def _call_llm_summarize(text: str, max_tokens: int = 120) -> str:
    """Use the memory_system LLM; fallback to short truncation summarizer when LLM fails."""
    try:
        llm = getattr(memory_system, "llm_controller", None)
        if llm and getattr(llm, "llm", None):
            # adapt to your llm controller API already used in file
            prompt = f"Summarize briefly (max {max_tokens} tokens):\n\n{text}"
            try:
                # preferred interface used elsewhere in your file:
                response = memory_system.llm_controller.llm.get_completion(prompt)
            except Exception:
                # some controllers accept call like llm(prompt)
                response = memory_system.llm_controller.llm(prompt)
            # ensure string
            if isinstance(response, dict):
                # some clients return {'text': '...'}
                return response.get("text") or response.get("content") or str(response)
            return str(response)
    except Exception as e:
        # fall back
        pass

    # fallback simple summarization (safe, non-LLM)
    text = text.strip()
    if not text:
        return ""
    if len(text) > 400:
        return text[:400].rsplit('.', 1)[0] + "..."
    return text


# --- Initialize AgenticMemorySystem ---
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend='openai',
    llm_model='gpt-4o-mini',
    evo_threshold=100,
    api_key=OPENAI_API_KEY,
    search_endpoint=AZURE_SEARCH_ENDPOINT,
    search_key=AZURE_SEARCH_KEY,
    search_index=AZURE_SEARCH_INDEX
)

# --- API Endpoints ---

@app.post("/optimize/onboarding_to_matching")
async def optimize_onboarding_to_matching(payload: OptimizeOnboardToMatchRequest = Body(...)):
    """
    Return a compact, matcher-friendly package from onboarding memories:
    - Returns requested selectors verbatim from top results
    - Summarizes the rest
    - Optionally filters to grant-type memories and discards irrelevant grants
    """
    try:
        # Build search query (session filter shorthand)
        q = payload.query
        if payload.session_filter:
            # memory_system.search understands "session_key:XYZ" style filters
            q = f"session_key:{payload.session_filter} {q}"

        # Retrieve top-k candidates
        search_out = memory_system.search(q, payload.top_k)
        metas = []
        if search_out and "metadatas" in search_out and search_out["metadatas"]:
            metas = search_out["metadatas"][0]
        else:
            metas = []

        # If user wants only grants, filter by memory_type/category/tags
        candidates = []
        for m in metas:
            # normalize keys
            m_type = (m.get("memory_type") or m.get("category") or "").lower()
            tags = [t.lower() for t in (m.get("tags") or [])] if m.get("tags") else []
            is_grant = ("grant" in m_type) or ("grant" in tags) or ("grant" in (m.get("keywords") or []))
            # compute a simple relevance score: if distances present use them; else fallback to search score heuristics
            score = None
            # memory_system.search returns distances in search_out["distances"]
            try:
                idx = metas.index(m)
                dists = search_out.get("distances", [[]])[0]
                score = 1.0 - dists[idx] if dists and idx < len(dists) else None
            except Exception:
                score = None

            if payload.grant_only:
                if not is_grant:
                    # discard non-grants
                    continue
                # if we have a score threshold, respect it
                if score is not None and score < payload.grant_relevance_threshold:
                    # discard low relevance grants
                    continue

            # keep candidate
            candidates.append({**m, "_score": score})

        # Build selected outputs (selectors are keys to keep, e.g. 'customer' or 'intent')
        selected = []
        omitted_texts = []
        for c in candidates:
            # if selectors provided, pull them into a small dict
            if payload.selectors:
                sel = {}
                for key in payload.selectors:
                    if key in c:
                        sel[key] = c[key]
                selected.append({"id": c.get("id"), "selected": sel, "_meta": {k: c.get(k) for k in ("memory_type","timestamp","_score")}})
                # save the non-selected content to omitted for summary
                omitted_texts.append(c.get("content", "") if c.get("content") else json.dumps({k:v for k,v in c.items() if k not in payload.selectors}))
            else:
                # no selectors specified: keep small canonical metadata and content
                selected.append({"id": c.get("id"), "content": c.get("content"), "_meta": {k: c.get(k) for k in ("memory_type","timestamp","_score")}})
                # but also append full content for summarization if needed
                omitted_texts.append(c.get("content", ""))

        # Summarize omitted parts if requested
        summary = None
        if payload.summarize:
            merged = "\n\n".join([t for t in omitted_texts if t])
            summary = _call_llm_summarize(merged, payload.summary_max_tokens)

        meta = {
            "requested_query": payload.query,
            "session_filter": payload.session_filter,
            "returned_count": len(selected),
            "original_candidates": len(metas)
        }

        return {"status":"ok", "selected": selected, "summary": summary, "meta": meta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Endpoint: Matching -> Onboarding ---
@app.post("/optimize/matching_to_onboarding")
async def optimize_matching_to_onboarding(payload: OptimizeMatchToOnboardRequest = Body(...)):
    """
    Return onboarding-friendly compact package for a matching agent's needs:
    - returns requested selectors verbatim (e.g. contact info, instructions)
    - summarizes the rest so onboarding sees concise context
    """
    try:
        q = payload.query
        if payload.session_filter:
            q = f"session_key:{payload.session_filter} {q}"

        search_out = memory_system.search(q, payload.top_k)
        metas = search_out.get("metadatas", [[]])[0] if search_out else []

        selected = []
        omitted_texts = []
        for m in metas:
            # keep exact fields if requested
            if payload.selectors:
                sel = {k: m.get(k) for k in payload.selectors if k in m}
                selected.append({"id": m.get("id"), "selected": sel, "_meta": {k: m.get(k) for k in ("memory_type","timestamp")}})
                omitted_texts.append(m.get("content", ""))
            else:
                # default: keep short metadata + content snippet
                selected.append({"id": m.get("id"), "content_snippet": (m.get("content") or "")[:400], "_meta": {k: m.get(k) for k in ("memory_type","timestamp")}})
                omitted_texts.append(m.get("content", ""))

        summary = None
        if payload.summarize:
            merged = "\n\n".join([t for t in omitted_texts if t])
            summary = _call_llm_summarize(merged, payload.summary_max_tokens)

        meta = {
            "requested_query": payload.query,
            "returned_count": len(selected),
            "original_candidates": len(metas)
        }
        return {"status":"ok", "selected": selected, "summary": summary, "meta": meta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Check service health and search connectivity."""
    try:
        # Test search connectivity
        memory_system.retriever.search_client.get_document_count()
        return {
            "status": "healthy",
            "search_service": "connected",
            "index": AZURE_SEARCH_INDEX
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue a JWT access token."""
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": AUTH_USERNAME})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/memory/longterm")
async def write_longterm_memory(
    item: MemoryPayload,
    current_user: str = Depends(get_current_user)
):
    """Save a memory note with idempotency check and blob storage."""
    try:
        return memory_system.write_longterm_memory(
            session_key=item.session_key,
            user_id=item.user_id,
            memory_type=item.memory_type,
            payload=item.payload,
            request_id=item.request_id
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/session/{session_key}")
async def get_session(
    session_key: str,
    k: int = Query(100, description="Max number of results"),
    current_user: str = Depends(get_current_user)
):
    """Retrieve the most recent memory for a session key."""
    try:
        return memory_system.get_session(session_key, k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.patch("/memory/patch")
async def patch_memory(
    session_key: str = Query(...),
    user_id: Optional[str] = Query(None),
    patch: Dict[str, Any] = Body(...),
    request_id: Optional[str] = Query(None),
    expected_version: Optional[int] = Query(None),
    current_user: str = Depends(get_current_user)
):
    """Update a memory with deep-merge patching and versioning."""
    try:
        return memory_system.patch_memory(
            session_key=session_key,
            user_id=user_id,
            patch=patch,
            request_id=request_id,
            expected_version=expected_version
        )
    except ValueError as e:
        if "Version mismatch" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/delete/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: str = Depends(get_current_user)
):
    """Delete a memory note by ID."""
    try:
        return memory_system.delete_memory(memory_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/memory/{memory_id}")
async def get_memory(
    memory_id: str,
    current_user: str = Depends(get_current_user)
):
    """Retrieve a specific memory by ID."""
    try:
        memory = memory_system.read(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Handle blob URLs
        if memory.get("blob_url"):
            try:
                payload = memory_system._fetch_payload_from_blob(memory["blob_url"])
                memory["payload"] = payload
                memory["content"] = f"[Blob content: {memory.get('size_bytes', 0)} bytes]"
            except Exception as e:
                logger.warning(f"Could not fetch blob: {e}")
        
        return {
            "status": "ok",
            "memory": memory
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
async def search_memories_post(
    query_data: SearchQuery,
    current_user: str = Depends(get_current_user)
):
    """
    Search memories with optional neighbor inclusion.
    This is the main RAG retrieval endpoint.
    """
    try:
        if query_data.include_neighbors:
            results = memory_system.search_with_neighbors(query_data.query, query_data.k)
        else:
            results = memory_system.search(query_data.query, query_data.k)
        
        return {
            "status": "ok",
            "query": query_data.query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_memories_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results"),
    include_neighbors: bool = Query(False, description="Include linked memories"),
    current_user: str = Depends(get_current_user)
):
    """
    Search memories (GET version for simple queries).
    This exposes the RAG functionality.
    """
    try:
        if include_neighbors:
            results = memory_system.search_with_neighbors(query, k)
        else:
            results = memory_system.search(query, k)
        
        return {
            "status": "ok",
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/related/{memory_id}")
async def get_related_memories(
    memory_id: str,
    k: int = Query(5, description="Number of related memories"),
    current_user: str = Depends(get_current_user)
):
    """Get memories related to a specific memory (via links or similarity)."""
    try:
        source = memory_system.read(memory_id)
        if not source:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Search for similar memories
        results = memory_system.search(source.get("content", ""), k=k+1)
        
        # Exclude the source memory itself
        related = [r for r in results if r.get("id") != memory_id][:k]
        
        return {
            "status": "ok",
            "source_id": memory_id,
            "count": len(related),
            "related_memories": related
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/rag/query")
async def rag_query(
    query: str = Body(..., embed=True),
    k: int = Body(5, embed=True),
    include_neighbors: bool = Body(False, embed=True),
    current_user: str = Depends(get_current_user)
):
    """
    Full RAG endpoint: Retrieve relevant memories and generate response using LLM.
    This is the complete RAG implementation.
    """
    try:
        # 1. Retrieve relevant memories
        if include_neighbors:
            memories = memory_system.search_with_neighbors(query, k)
        else:
            memories = memory_system.search(query, k)
        
        if not memories:
            return {
                "status": "ok",
                "query": query,
                "answer": "No relevant memories found to answer this query.",
                "sources": [],
                "retrieved_count": 0
            }
        
        # 2. Format context from retrieved memories
        context_parts = []
        for i, mem in enumerate(memories):
            content = mem.get('content', '')
            
            # Handle blob URLs
            if mem.get('blob_url') and content.startswith("Payload stored in blob:"):
                try:
                    payload = memory_system._fetch_payload_from_blob(mem['blob_url'])
                    content = json.dumps(payload)[:500] + "..."
                except Exception:
                    content = "[Large blob content]"
            
            context_parts.append(
                f"Memory {i+1} (ID: {mem.get('id', 'unknown')}):\n"
                f"Content: {content[:300]}...\n"
                f"Context: {mem.get('context', 'N/A')}\n"
                f"Keywords: {', '.join(mem.get('keywords', []))}\n"
                f"Tags: {', '.join(mem.get('tags', []))}"
            )
        
        context = "\n\n".join(context_parts)
        
        # 3. Generate response using LLM with retrieved context
        prompt = f"""Based on the following memories from the knowledge base, answer the query.

Retrieved Memories:
{context}

Query: {query}

Instructions:
- Use information from the memories to answer the query
- If the memories don't contain relevant information, say so
- Cite which memory number you're referencing
- Be concise and accurate

Answer:"""
        
        try:
            #response = memory_system.llm_controller.llm.get_completion(prompt)
            try:
                response = memory_system.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {"type": "array", "items": {"type": "string"}},
                                "context": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }}
                )
            except TypeError:
                # Approach 2: Try without response_format

                try:
                    response = memory_system.llm_controller.llm.get_completion(prompt)
                except TypeError:
                    # Approach 3: Try as direct call if it's a callable
                    if callable(memory_system.llm_controller.llm):
                        response = memory_system.llm_controller.llm(prompt)
                    else:
                        raise Exception("Unable to determine LLM call signature")
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            response = "Error generating response from LLM: {str(e)}."
        
        return {
            "status": "ok",
            "query": query,
            "answer": response,
            "sources": [m.get("id") for m in memories],
            "retrieved_count": len(memories),
            "context_used": len(context)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/consolidate")
async def trigger_consolidation(current_user: str = Depends(get_current_user)):
    """
    Manually trigger memory consolidation/evolution.
    This can be used to periodically optimize memory relationships.
    """
    try:
        # This would trigger batch evolution processing
        # For now, it's a placeholder
        return {
            "status": "ok",
            "message": "Consolidation triggered",
            "note": "Background processing initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#@app.get("/memory/stats")
#async def get_memory_stats(current_user: str = Depends(get_current_user)):
    """Get statistics about the memory system."""
#    try:
        # Use get_document_count() instead of trying to get a document called 'stats'
#        from azure.core.exceptions import ResourceNotFoundError
        
#        try:
#            doc_count = memory_system.retriever.search_client.get_document_count()
#        except Exception as e:
#            logger.error(f"Failed to get document count: {e}")
#            doc_count = 0
        
#        return {
#            "status": "ok",
#            "total_memories": doc_count,
#            "index_name": AZURE_SEARCH_INDEX,
#            "search_endpoint": AZURE_SEARCH_ENDPOINT
#        }
#    except Exception as e:
#        logger.error(f"Error getting stats: {e}")
#        raise HTTPException(status_code=500, detail=str(e))

#@app.get("/memory/stats")
#async def get_memory_stats(current_user: str = Depends(get_current_user)):
    """Get statistics about the memory system."""
#    try:
#        doc_count = memory_system.retriever.search_client.get_document_count()
        
#        return {
#            "status": "ok",
#            "total_memories": doc_count,
#            "index_name": AZURE_SEARCH_INDEX,
#            "search_endpoint": AZURE_SEARCH_ENDPOINT
#        }
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
@app.get("/debug/links/{memory_id}")
async def debug_memory_links(
    memory_id: str,
    current_user: str = Depends(get_current_user)
):
    """Debug endpoint to check why links aren't being created."""
    try:
        # Get the memory
        memory = memory_system.read(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Search for similar memories
        similar = memory_system.search(memory.get("content", ""), k=5)
        
        # Calculate what links SHOULD have been created
        debug_info = {
            "memory_id": memory_id,
            "current_links": memory.get("links", []),
            "content_preview": memory.get("content", "")[:200],
            "similar_memories": []
        }
        
        for mem in similar:
            distance = 1.0 - mem.get("score", 1.0)
            similarity = 1.0 - mem.get("similarity", distance)
            
            should_link = distance < 0.3
            
            debug_info["similar_memories"].append({
                "id": mem.get("id"),
                "distance": distance,
                "similarity": similarity,
                "should_auto_link": should_link,
                "reason": f"distance {distance:.4f} < 0.3" if should_link else f"distance {distance:.4f} >= 0.3",
                "content_preview": mem.get("content", "")[:100]
            })
        
        return {
            "status": "ok",
            "debug": debug_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)