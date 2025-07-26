#!/usr/bin/env python3
"""
Test script for database connection and patient data retrieval.
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.database_service import DatabaseService
from app.services.vectorization_service import VectorizationService

async def test_database_connection():
    """Test database connection and patient data retrieval."""
    print("🔍 Testing database connection...")
    
    try:
        # Test database service
        db_service = DatabaseService()
        print("✅ Database service initialized successfully")
        
        # Test health check
        health = db_service.check_database_health()
        print(f"📊 Database health: {health}")
        
        # Get all patients
        patients = db_service.get_all_patients()
        print(f"👥 Found {len(patients)} patients in database")
        
        # Show first few patients (raw data)
        if patients:
            print("\n📋 Sample patient data (raw):")
            for i, patient in enumerate(patients[:3]):
                print(f"  {i+1}. {patient}")
        
        # Convert to natural language
        descriptions = db_service.convert_patients_to_natural_language(patients)
        print(f"\n📝 Converted {len(descriptions)} patients to natural language")
        
        # Show natural language descriptions
        if descriptions:
            print("\n🗣️  Sample patient descriptions (natural language):")
            for i, desc in enumerate(descriptions[:3]):
                print(f"  {i+1}. {desc}")
        
        return patients, descriptions
        
    except Exception as e:
        print(f"❌ Error testing database: {e}")
        raise

async def test_vectorization_service():
    """Test vectorization service with patient data."""
    print("\n🚀 Testing vectorization service...")
    
    try:
        # Initialize vectorization service
        vectorization_service = VectorizationService()
        print("✅ Vectorization service initialized successfully")
        
        # Check health
        health = vectorization_service.check_health()
        print(f"📊 Vectorization service health: {health}")
        
        # Get patient data summary
        summary = await vectorization_service.get_patient_data_summary()
        print(f"📈 Patient data summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vectorization service: {e}")
        return False

async def main():
    """Main test function."""
    print("🏥 MedBot Assistant - Database and Vectorization Test")
    print("=" * 60)
    
    try:
        # Test database connection
        patients, descriptions = await test_database_connection()
        
        # Test vectorization service
        vectorization_success = await test_vectorization_service()
        
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        print(f"   • Database connection: ✅ Success")
        print(f"   • Patients retrieved: {len(patients) if patients else 0}")
        print(f"   • Natural language conversion: ✅ Success")
        print(f"   • Vectorization service: {'✅ Success' if vectorization_success else '❌ Failed'}")
        
        if patients and descriptions:
            print(f"\n🎉 All tests passed! Ready to use the API with {len(patients)} patients.")
        else:
            print("\n⚠️  Tests completed but no patient data found.")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
