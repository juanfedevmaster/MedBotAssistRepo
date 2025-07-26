# MedBot Assistant API

API para asistente médico con capacidades de vectorización y agentes IA.

## Estructura del Proyecto

```
MedBotAssist.BotOpenIA/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── vectorization.py    # Endpoints de vectorización
│   │   └── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Configuración de la aplicación
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Modelos Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   └── vectorization_service.py  # Lógica de vectorización
│   └── __init__.py
├── main.py                        # Aplicación FastAPI principal
├── run_server.py                  # Script para ejecutar el servidor
├── requirements.txt               # Dependencias
├── .env.example                   # Ejemplo de variables de entorno
└── README.md                      # Este archivo
```

## Configuración

1. **Copia el archivo de variables de entorno:**
   ```bash
   copy .env.example .env
   ```

2. **Configura tu API key de OpenAI en el archivo `.env`:**
   ```
   OPENAI_API_KEY=tu_api_key_aqui
   ```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

### Opción 1: Usando el script
```bash
python run_server.py
```

### Opción 2: Usando uvicorn directamente
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints Disponibles

### 📊 Documentación Automática
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 🏥 Endpoints Principales

#### Health Check
- **GET** `/` - Verificación básica
- **GET** `/health` - Verificación detallada

#### Vectorización y Pacientes
- **POST** `/api/v1/vectorization/search` - Buscar pacientes similares usando vectorización
- **POST** `/api/v1/vectorization/health` - Estado del servicio de vectorización y base de datos
- **GET** `/api/v1/vectorization/collections` - Listar colecciones vectoriales disponibles
- **GET** `/api/v1/vectorization/patients/summary` - Resumen de datos de pacientes desde SQL Server

## Pruebas

### Probar conexión a base de datos:
```bash
python test_database.py
```

Este script:
- ✅ Prueba la conexión a SQL Server
- ✅ Consulta la tabla Patients
- ✅ Convierte datos a lenguaje natural
- ✅ Inicializa el servicio de vectorización
- ✅ Verifica el estado de todos los componentes

## Ejemplo de Uso

### Buscar pacientes similares:
```json
POST /api/v1/vectorization/search
{
  "query": "paciente con diabetes que necesita seguimiento",
  "top_k": 5,
  "similarity_threshold": 0.7,
  "include_metadata": true
}
```

### Respuesta esperada:
```json
{
  "query": "paciente con diabetes que necesita seguimiento",
  "embedding_model": "text-embedding-3-small",
  "documents": [
    {
      "id": "patient_0",
      "content": "Paciente Juan Pérez con número de identificación 12345678, 45 años de edad, nacido el 15 de marzo de 1978, teléfono de contacto 555-0123, correo electrónico juan.perez@email.com.",
      "similarity_score": 0.85,
      "metadata": {
        "type": "patient_description",
        "index": 0
      }
    }
  ],
  "total_documents": 150,
  "search_time_ms": 245.2,
  "timestamp": "2025-07-25T10:30:00.000Z"
}
```

### Obtener resumen de pacientes:
```bash
GET /api/v1/vectorization/patients/summary
```

## Funcionalidades Principales

✅ **Consulta automática a SQL Server** - Conecta y consulta la tabla Patients
✅ **Conversión a lenguaje natural** - Transforma datos estructurados en descripciones naturales
✅ **Vectorización automática** - Genera embeddings de las descripciones de pacientes
✅ **Búsqueda semántica** - Encuentra pacientes similares usando vectores
✅ **Almacenamiento en ChromaDB** - Guarda vectores para búsquedas rápidas
✅ **API REST completa** - Endpoints documentados y validados
✅ **Namespaces especializados** - Organización por tipos de datos médicos

## Namespace Demográfico: `demographic_patients_namespace`

### 🎯 Descripción
Implementación especializada para la organización y búsqueda de datos demográficos de pacientes mediante vectorización enfocada.

### 📊 Características del Namespace Demográfico

- **Datos Enfocados**: Solo información demográfica (edad, sexo, tipo sanguíneo, fecha de creación)
- **Optimización**: Vectorización específica para búsquedas demográficas
- **Separación**: Colección independiente `demographic_patients_namespace` en ChromaDB
- **Metadatos Enriquecidos**: Incluye namespace, timestamps y categorías específicas

### 🚀 Ventajas del Namespace Demográfico

- 🎯 **Búsquedas Precisas**: Enfoque específico en características demográficas
- ⚡ **Rendimiento Optimizado**: Vectores especializados para consultas demográficas
- 🔍 **Filtrado Avanzado**: Separación clara entre datos demográficos y clínicos
- 📈 **Escalabilidad**: Fácil extensión para otros namespaces especializados

### ⚙️ Configuración del Namespace

```python
# En config.py
CHROMA_DEMOGRAPHIC_COLLECTION = "demographic_patients_namespace"

# Metadatos específicos incluidos:
{
    "namespace": "demographic_patients_namespace",
    "type": "patient_demographic_data",
    "age": patient.age,
    "gender": patient.gender,
    "blood_type": patient.blood_type,
    "vectorized_at": timestamp
}
```

### 🔍 Ejemplo de Búsqueda Demográfica

```json
POST /api/v1/vectorization/search
{
  "query": "paciente masculino de 45 años con tipo de sangre O+",
  "top_k": 5,
  "similarity_threshold": 0.7,
  "include_metadata": true
}
```

**Respuesta con namespace demográfico:**
```json
{
  "query": "paciente masculino de 45 años con tipo de sangre O+",
  "embedding_model": "text-embedding-3-small",
  "documents": [
    {
      "id": "demo_patient_123",
      "content": "Paciente masculino de 45 años con tipo de sangre O+...",
      "similarity_score": 0.95,
      "metadata": {
        "namespace": "demographic_patients_namespace",
        "type": "patient_demographic_data",
        "age": 45,
        "gender": "Masculino",
        "blood_type": "O+",
        "collection_used": "demographic_patients_namespace"
      }
    }
  ],
  "total_documents": 6,
  "search_time_ms": 245
}
```

### 🔄 Vectorización Incremental Demográfica

El sistema detecta automáticamente nuevos pacientes y vectoriza solo la información demográfica nueva en el namespace específico:

```python
# Vectorización automática incremental
result = await vectorization_service.vectorize_new_patient_data()
# Solo vectoriza datos demográficos nuevos en demographic_patients_namespace
```

## Características

✅ **Estructura modular** con separación de responsabilidades
✅ **Validación automática** con Pydantic
✅ **Documentación automática** con Swagger/OpenAPI
✅ **Configuración centralizada** con variables de entorno
✅ **Manejo de errores** robusto
✅ **Logging** configurado
✅ **CORS** habilitado
✅ **Health checks** implementados

## Próximos Pasos

Los endpoints están listos para implementar la lógica de:

1. **Generación de embeddings** con OpenAI
2. **Búsqueda de similitud** en ChromaDB
3. **Procesamiento de resultados**
4. **Health checks** de componentes

## Tecnologías

- **FastAPI** - Framework web moderno
- **Pydantic** - Validación de datos
- **OpenAI** - Embeddings y LLM
- **ChromaDB** - Base de datos vectorial
- **LangChain** - Framework para LLM
- **LangGraph** - Flujos multiagente

## Comandos de Soporte

-- **Limpiar ChromaDB** - python clear_patients.py
