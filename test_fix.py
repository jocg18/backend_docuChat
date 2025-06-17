#!/usr/bin/env python3

import requests
import json

def test_upload_and_query():
    """Prueba el upload y query con los cambios de detección de PDF"""
    
    # Test con el file_id que ya sabemos que funciona
    file_id = "6e7a9510-6f44-464c-acd5-3e80e062c328"
    
    print("🧪 Probando query con detección corregida de PDFs...")
    
    # Test query
    query_data = {
        "query": "¿De qué trata este documento?",
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
            print(f"📄 Respuesta: {result.get('answer', 'No answer')[:200]}...")
            
            # Verificar si ahora dice "PDF" en lugar de "imagen"
            answer = result.get('answer', '')
            if '📄' in answer and 'PDF' in answer:
                print("🎉 ¡ÉXITO! Ahora detecta correctamente los PDFs")
            elif '🖼️' in answer and 'imagen' in answer:
                print("⚠️  Aún muestra como imagen, el cambio no se aplicó")
            else:
                print("🤔 Respuesta diferente:", answer[:100])
                
        else:
            print(f"❌ Error en query: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_upload_and_query()

