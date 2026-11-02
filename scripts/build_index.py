"""
Tahap 2 - Pembangunan Indeks Vektor.
Menjalankan build/rebuild indeks FAISS dari dataset basis pengetahuan.
"""
import sys
import os

# Tambahkan direktori induk ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import build_vector_store


def main():
    print("="*60)
    print("BUILDING VECTOR STORE")
    print("="*60)
    print()
    
    try:
        store = build_vector_store()
        
        print()
        print("="*60)
        print("✅ Vector store built successfully!")
        print(f"   Total vectors: {store.index.ntotal}")
        print("="*60)
        
        # Uji pencarian
        print("\n📋 Testing search functionality...")
        test_query = "deyen poi ka kampuong"
        results = store.search(test_query, top_k=3)
        
        print(f"\nTest query: '{test_query}'")
        print("Results:")
        for entry, score in results:
            print(f"  [{score:.3f}] {entry.bahasa_ocu} → {entry.bahasa_indonesia}")
            
        print("\n✅ Search test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
