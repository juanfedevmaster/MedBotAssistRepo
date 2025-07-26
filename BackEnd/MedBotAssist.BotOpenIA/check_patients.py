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
    
    print(f'Número total de pacientes vectorizados: {len(data["ids"]) if data["ids"] else 0}')
    
    if data['ids']:
        print('IDs de pacientes:')
        for i, patient_id in enumerate(data['ids']):
            print(f'  {i+1}. {patient_id}')
            if i < 3 and data['documents']:  # Show first 3 descriptions
                print(f'     Descripción: {data["documents"][i][:100]}...')
    
except Exception as e:
    print(f'Error: {e}')
