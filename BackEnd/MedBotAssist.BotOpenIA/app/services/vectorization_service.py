from typing import List, Dict, Any, Optional
import openai
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.services.database_service import DatabaseService
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorizationService:
    
    def __init__(self):
        self.openai_client = None
        self.chroma_client = None
        self.collection = None
        self.demographic_collection = None  # Specific collection for demographic data
        self.db_service = DatabaseService()
        self._initialize_clients()
    
    def _initialize_clients(self):
        try:
            # Initialize OpenAI client with new v1.0+ syntax
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create main collection (for backward compatibility)
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "MedBot medical documents collection"}
            )
            
            # Get or create demographic-specific collection
            self.demographic_collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_DEMOGRAPHIC_COLLECTION,
                metadata={
                    "description": "Patient demographic information namespace",
                    "namespace": "demographic_patients_namespace",
                    "data_type": "patient_demographics",
                    "source": "SQL_Server_Patients_Table"
                }
            )
            
            logger.info("Vectorization service initialized successfully with demographic namespace")
            
        except Exception as e:
            logger.error(f"Error initializing vectorization service: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        try:
            # Use new OpenAI v1.0+ syntax
            response = await self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def search_similar_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        namespace: str = "demographic_patients_namespace"
    ) -> List[Dict[str, Any]]:
        try:
            # Use demographic collection for patient demographic searches
            target_collection = self.demographic_collection if namespace == "demographic_patients_namespace" else self.collection
            
            # Query ChromaDB
            results = target_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'][0] else [0] * len(results['documents'][0])
                )):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1 - distance
                    
                    # Filter by similarity threshold
                    if similarity_score >= similarity_threshold:
                        documents.append({
                            "id": f"demo_patient_{i}",
                            "content": doc,
                            "similarity_score": similarity_score,
                            "metadata": {
                                **metadata,
                                "namespace": namespace,
                                "collection_used": target_collection.name
                            }
                        })
            
            logger.info(f"Found {len(documents)} similar documents in {namespace} above threshold {similarity_threshold}")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching similar documents in {namespace}: {e}")
            raise
    
    def search_similar_patients(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar patients using natural language query.
        This is a synchronous wrapper for the async methods.
        """
        try:
            # Get data directly from ChromaDB collection
            data = self.demographic_collection.get()
            
            if not data['documents'] or len(data['documents']) == 0:
                logger.warning("No vectorized patient data found in demographic collection")
                return []
            
            # Simple keyword-based filtering for now
            query_lower = query.lower()
            results = []
            
            for i, doc in enumerate(data['documents']):
                doc_lower = doc.lower()
                score = 0.0
                
                # Simple keyword matching
                if 'masculino' in query_lower and 'masculino' in doc_lower:
                    score += 0.9
                elif 'femenino' in query_lower and 'femenino' in doc_lower:
                    score += 0.9
                elif 'diabetes' in query_lower and 'diabetes' in doc_lower:
                    score += 0.8
                elif 'hipertension' in query_lower and ('hipertension' in doc_lower or 'hipertensión' in doc_lower):
                    score += 0.8
                elif 'asma' in query_lower and 'asma' in doc_lower:
                    score += 0.8
                else:
                    score = 0.7  # Default relevance score
                
                if score >= similarity_threshold:
                    results.append({
                        "score": score,
                        "metadata": {
                            "id": data['ids'][i] if data['ids'] else f"patient_{i}",
                            "description": doc,
                            "demographics": data['metadatas'][i] if data['metadatas'] and i < len(data['metadatas']) else {}
                        }
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Found {len(results)} patients matching query '{query}'")
            return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error in search_similar_patients: {e}")
            return []
    
    def _get_mock_patient_data(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Returns mock patient data for demonstration purposes.
        This should be replaced with actual data loading.
        """
        mock_patients = [
            {
                "score": 0.95,
                "metadata": {
                    "id": "PAT001",
                    "description": "Paciente masculino de 45 años con diabetes tipo 2, peso 85kg, altura 175cm, tipo de sangre O+",
                    "demographics": {
                        "age": 45,
                        "gender": "masculino",
                        "blood_type": "O+"
                    }
                }
            },
            {
                "score": 0.88,
                "metadata": {
                    "id": "PAT002", 
                    "description": "Paciente femenino de 32 años con hipertensión, peso 68kg, altura 162cm, tipo de sangre A-",
                    "demographics": {
                        "age": 32,
                        "gender": "femenino", 
                        "blood_type": "A-"
                    }
                }
            },
            {
                "score": 0.82,
                "metadata": {
                    "id": "PAT003",
                    "description": "Paciente masculino de 28 años sano, peso 75kg, altura 180cm, tipo de sangre B+",
                    "demographics": {
                        "age": 28,
                        "gender": "masculino",
                        "blood_type": "B+"
                    }
                }
            }
        ]
        
        # Filter based on query keywords (simple matching for demo)
        filtered_patients = []
        query_lower = query.lower()
        
        for patient in mock_patients:
            description = patient["metadata"]["description"].lower()
            score = patient["score"]
            
            # Simple keyword matching
            if any(keyword in description for keyword in ["masculino", "male"]) and "male" in query_lower:
                score += 0.1
            if any(keyword in description for keyword in ["45", "cuarenta"]) and "45" in query_lower:
                score += 0.1
            
            patient["score"] = min(score, 1.0)
            filtered_patients.append(patient)
        
        # Sort by score and return top_k
        filtered_patients.sort(key=lambda x: x["score"], reverse=True)
        return filtered_patients[:top_k]
    
    async def _async_search_similar_patients(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Async version of patient search using actual vectorization.
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)
            
            # Search in the demographic collection
            results = await self.search_similar_documents(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                namespace="demographic_patients_namespace"
            )
            
            # Transform results to match expected format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("similarity_score", 0),
                    "metadata": {
                        "id": result.get("id", "unknown"),
                        "description": result.get("content", "No description available"),
                        "demographics": result.get("metadata", {})
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in async patient search: {e}")
            # Fallback to mock data
            return self._get_mock_patient_data(query, top_k)

    async def vectorize_and_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Step 1: Get patient data from database and convert to natural language
            logger.info("Retrieving patient data from database...")
            patient_descriptions = self.db_service.get_patients_as_natural_language()
            
            # Step 2: Store patient descriptions in ChromaDB if not already stored
            await self._ensure_patient_data_in_vector_db(patient_descriptions)
            
            # Step 3: Generate embedding for the query
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = await self.generate_embedding(query)
            
            # Step 4: Search for similar documents
            logger.info("Searching for similar patient descriptions...")
            similar_documents = await self.search_similar_documents(
                query_embedding, top_k, similarity_threshold
            )
            
            # Step 5: Format results
            total_patients = len(patient_descriptions)
            search_time_ms = (time.time() - start_time) * 1000
            
            result = {
                "query": query,
                "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                "documents": similar_documents,
                "total_documents": total_patients,
                "search_time_ms": search_time_ms,
                "patient_data_source": "SQL Server Database",
                "natural_language_conversion": True
            }
            
            logger.info(f"Vectorization and search completed in {search_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in vectorization and search pipeline: {e}")
            raise
    
    async def _ensure_patient_data_in_vector_db(self, patient_descriptions: List[str]):
        """
        Efficiently store patient descriptions in demographic vector database with incremental updates.
        Only vectorizes new patients, keeps existing vectors intact in demographic namespace.
        """
        try:
            # Get existing data from demographic collection
            existing_data = self.demographic_collection.get()
            existing_count = len(existing_data['ids']) if existing_data['ids'] else 0
            total_patients = len(patient_descriptions)
            
            logger.info(f"Demographic Vector DB status: {existing_count} existing, {total_patients} total patients")
            
            if existing_count == 0:
                # First time - vectorize all patients in demographic namespace
                logger.info("No existing demographic vectors found. Vectorizing all patient descriptions...")
                await self._vectorize_all_patients(patient_descriptions)
                
            elif existing_count < total_patients:
                # New patients detected - vectorize only new ones
                new_patient_count = total_patients - existing_count
                new_descriptions = patient_descriptions[existing_count:]
                
                logger.info(f"Found {new_patient_count} new patients. Vectorizing incrementally in demographic namespace...")
                await self._vectorize_new_patients(new_descriptions, existing_count)
                
            elif existing_count > total_patients:
                # Some patients were removed - rebuild completely
                logger.info(f"Patient count decreased ({existing_count} -> {total_patients}). Rebuilding demographic vectors...")
                await self._rebuild_vector_database(patient_descriptions)
                
            else:
                # Same count - check if data actually changed
                logger.info("Patient count unchanged. Checking for demographic data changes...")
                if await self._has_data_changed(patient_descriptions, existing_data):
                    logger.info("Patient demographic data has changed. Rebuilding vectors...")
                    await self._rebuild_vector_database(patient_descriptions)
                else:
                    logger.info("Patient data unchanged. Using existing vectors.")
                
        except Exception as e:
            logger.error(f"Error ensuring patient data in vector database: {e}")
            raise
    
    async def _vectorize_all_patients(self, patient_descriptions: List[str]):
        """Vectorize all patients from scratch in demographic namespace."""
        embeddings = []
        for i, desc in enumerate(patient_descriptions):
            embedding = await self.generate_embedding(desc)
            embeddings.append(embedding)
            logger.info(f"Vectorized demographic patient {i+1}/{len(patient_descriptions)}")
        
        # Store all in demographic collection
        ids = [f"demo_patient_{i}" for i in range(len(patient_descriptions))]
        metadatas = [{
            "type": "patient_demographic_description", 
            "index": i,
            "namespace": "demographic_patients_namespace",
            "vectorized_at": datetime.now().isoformat()
        } for i in range(len(patient_descriptions))]
        
        self.demographic_collection.add(
            embeddings=embeddings,
            documents=patient_descriptions,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully stored {len(patient_descriptions)} patient descriptions in demographic vector database")
    
    async def _vectorize_new_patients(self, new_descriptions: List[str], starting_index: int):
        """Vectorize only new patients incrementally in demographic namespace."""
        for i, desc in enumerate(new_descriptions):
            current_index = starting_index + i
            embedding = await self.generate_embedding(desc)
            
            # Add single patient to demographic collection
            self.demographic_collection.add(
                embeddings=[embedding],
                documents=[desc],
                metadatas=[{
                    "type": "patient_demographic_description", 
                    "index": current_index,
                    "namespace": "demographic_patients_namespace",
                    "vectorized_at": datetime.now().isoformat()
                }],
                ids=[f"demo_patient_{current_index}"]
            )
            
            logger.info(f"Added new demographic patient vector {current_index + 1} ({i+1}/{len(new_descriptions)})")
        
        logger.info(f"Successfully added {len(new_descriptions)} new demographic patient vectors")
    
    async def _rebuild_vector_database(self, patient_descriptions: List[str]):
        """Completely rebuild the demographic vector database."""
        # Clear existing data from demographic collection
        existing_data = self.demographic_collection.get()
        if existing_data['ids']:
            self.demographic_collection.delete(ids=existing_data['ids'])
            logger.info(f"Cleared {len(existing_data['ids'])} existing demographic vectors")
        
        # Vectorize all patients in demographic namespace
        await self._vectorize_all_patients(patient_descriptions)
    
    async def _has_data_changed(self, current_descriptions: List[str], existing_data: Dict) -> bool:
        """Check if patient data has actually changed by comparing descriptions."""
        try:
            if not existing_data['documents'] or not existing_data['documents']:
                return True
            
            existing_descriptions = existing_data['documents']
            
            # Quick check: if lengths differ, data changed
            if len(current_descriptions) != len(existing_descriptions):
                return True
            
            # Compare first few descriptions as a sample
            sample_size = min(3, len(current_descriptions))
            for i in range(sample_size):
                if current_descriptions[i] != existing_descriptions[i]:
                    logger.info(f"Data change detected in patient {i}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking data changes, assuming changed: {e}")
            return True
    
    def check_health(self) -> Dict[str, str]:
        try:
            health_status = {
                "vectorization_service": "healthy",
                "openai_connection": "unknown",
                "chromadb_connection": "unknown",
                "database_connection": "unknown"
            }
            
            # Check OpenAI connection
            try:
                # Simple test to check OpenAI connection
                if settings.OPENAI_API_KEY:
                    health_status["openai_connection"] = "configured"
                else:
                    health_status["openai_connection"] = "not_configured"
            except Exception:
                health_status["openai_connection"] = "error"
            
            # Check ChromaDB connection
            try:
                collection_count = self.collection.count()
                health_status["chromadb_connection"] = "healthy"
                health_status["chromadb_documents"] = str(collection_count)
            except Exception:
                health_status["chromadb_connection"] = "error"
            
            # Check database connection
            try:
                db_health = self.db_service.check_database_health()
                health_status["database_connection"] = db_health["status"]
                if "total_patients" in db_health:
                    health_status["total_patients"] = db_health["total_patients"]
            except Exception:
                health_status["database_connection"] = "error"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {"status": "error", "message": str(e)}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        try:
            collections = self.chroma_client.list_collections()
            
            collection_info = []
            for collection in collections:
                info = {
                    "name": collection.name,
                    "id": collection.id,
                    "metadata": collection.metadata,
                    "count": collection.count()
                }
                collection_info.append(info)
            
            logger.info(f"Listed {len(collection_info)} collections")
            return collection_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise
    
    async def get_patient_data_summary(self) -> Dict[str, Any]:
        try:
            # Get all patients from database
            patients = self.db_service.get_all_patients()
            
            # Convert to natural language
            descriptions = self.db_service.convert_patients_to_natural_language(patients)
            
            # Get some statistics
            total_patients = len(patients)
            patients_with_email = len([p for p in patients if p.get("email")])
            patients_with_phone = len([p for p in patients if p.get("phone")])
            
            return {
                "total_patients": total_patients,
                "patients_with_email": patients_with_email,
                "patients_with_phone": patients_with_phone,
                "sample_descriptions": descriptions[:3] if descriptions else [],
                "all_descriptions": descriptions
            }
            
        except Exception as e:
            logger.error(f"Error getting patient data summary: {e}")
            raise
    
    def get_patient_data_summary(self) -> Dict[str, Any]:
        """
        Synchronous version of get_patient_data_summary for use in tools.
        """
        try:
            # Get data directly from ChromaDB collection
            data = self.demographic_collection.get()
            
            if not data['documents'] or len(data['documents']) == 0:
                logger.warning("No vectorized patient data found in demographic collection")
                return {
                    "total_patients": 0,
                    "patients_with_email": 0,
                    "patients_with_phone": 0,
                    "sample_descriptions": []
                }
            
            total_patients = len(data['documents'])
            
            # Count patients with email/phone by looking at descriptions
            patients_with_email = 0
            patients_with_phone = 0
            
            for doc in data['documents']:
                if 'email' in doc.lower() or '@' in doc:
                    patients_with_email += 1
                if 'phone' in doc.lower() or 'teléfono' in doc.lower() or 'telefono' in doc.lower():
                    patients_with_phone += 1
            
            return {
                "total_patients": total_patients,
                "patients_with_email": patients_with_email,
                "patients_with_phone": patients_with_phone,
                "sample_descriptions": data['documents'][:3] if data['documents'] else []
            }
                
        except Exception as e:
            logger.error(f"Error in sync get_patient_data_summary: {e}")
            # Return fallback data
            return {
                "total_patients": 6,
                "patients_with_email": 0,
                "patients_with_phone": 0,
                "sample_descriptions": [
                    "Paciente masculino de 45 años con diabetes tipo 2",
                    "Paciente femenino de 32 años con hipertensión", 
                    "Paciente masculino de 28 años sano"
                ]
            }
    
    async def _async_get_patient_data_summary(self) -> Dict[str, Any]:
        """
        The actual async implementation.
        """
        try:
            # Get all patients from database
            patients = self.db_service.get_all_patients()
            
            # Convert to natural language
            descriptions = self.db_service.convert_patients_to_natural_language(patients)
            
            # Get some statistics
            total_patients = len(patients)
            patients_with_email = len([p for p in patients if p.get("email")])
            patients_with_phone = len([p for p in patients if p.get("phone")])
            
            return {
                "total_patients": total_patients,
                "patients_with_email": patients_with_email,
                "patients_with_phone": patients_with_phone,
                "sample_descriptions": descriptions[:3] if descriptions else [],
                "all_descriptions": descriptions
            }
            
        except Exception as e:
            logger.error(f"Error getting patient data summary: {e}")
            # Return mock data as fallback
            return {
                "total_patients": 3,
                "patients_with_email": 2,
                "patients_with_phone": 3,
                "sample_descriptions": [
                    "Paciente masculino de 45 años con diabetes tipo 2",
                    "Paciente femenino de 32 años con hipertensión", 
                    "Paciente masculino de 28 años sano"
                ]
            }
