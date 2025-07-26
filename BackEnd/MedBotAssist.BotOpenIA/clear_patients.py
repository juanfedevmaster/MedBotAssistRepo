import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path='./chroma_db',
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Get the demographic collection
    collection = client.get_or_create_collection('demographic_patients_namespace')
    
    # Get all data
    data = collection.get()
    
    if data['ids']:
        print(f'Eliminando {len(data["ids"])} pacientes existentes...')
        collection.delete(ids=data['ids'])
        print('Base de datos vectorizada limpiada.')
    else:
        print('No hay pacientes para eliminar.')
    
    # Verify it's empty
    data_after = collection.get()
    print(f'Pacientes restantes: {len(data_after["ids"]) if data_after["ids"] else 0}')
    
except Exception as e:
    print(f'Error: {e}')
