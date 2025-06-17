#!/usr/bin/env python3

import requests
import json

def test_file_specific_query():
    """Prueba que las consultas usen el file_id específico del archivo subido"""
    
    # File ID conocido del upload anterior
    file_id = "6e7a9510-6f44-464c-acd5-3e80e062c328"
    
    print(f"🎯 Probando consulta específica con file_id: {file_id}")
    
    # Test query específica
    query_data = {
        "query": "¿Cómo se llama este documento?",
        "file_id": file_id
    }
    
    try:
        response = requests.post(
            "http://localhost:3000/api/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query exitosa!")
            print(f"📄 Tipo de consulta: {result.get('type', 'N/A')}")
            print(f"📋 Namespace usado: {result.get('namespace', 'N/A')}")
            print(f"📄 Respuesta: {result.get('answer', 'No answer')[:150]}...")
            
            # Verificar que está usando el namespace específico
            if result.get('namespace') == file_id:
                print("🎉 ¡PERFECTO! Está usando el file_id como namespace")
            else:
                print(f"⚠️  Error: namespace esperado {file_id}, obtenido {result.get('namespace')}")
                
            # Verificar que es una consulta específica, no general
            if result.get('type') == 'file_specific_query':
                print("✅ Tipo de consulta correcto: file_specific_query")
            else:
                print(f"⚠️  Tipo incorrecto: {result.get('type')} (debería ser file_specific_query)")
                
        else:
            print(f"❌ Error en query: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_debug_namespaces():
    """Verifica qué namespaces están disponibles"""
    print("\n🔍 Verificando namespaces disponibles...")
    
    try:
        response = requests.get("http://localhost:3000/api/debug/namespaces")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Total namespaces: {result.get('total_namespaces', 0)}")
            
            namespaces = result.get('namespaces', {})
            for namespace, info in namespaces.items():
                print(f"  - {namespace}: {info.get('vector_count', 0)} vectores")
        else:
            print(f"❌ Error obteniendo namespaces: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_file_specific_query()
    test_debug_namespaces()

