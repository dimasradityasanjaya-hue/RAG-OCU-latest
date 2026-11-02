"""
Komponen pendukung aplikasi - Antarmuka Baris Perintah (CLI).
Menyediakan penggunaan interaktif dan terjemahan langsung via terminal.
"""
import sys
import argparse

from src.translator import create_translator


def print_banner():
    """Menampilkan spanduk aplikasi."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          SISTEM PENERJEMAHAN BAHASA OCU                     ║")
    print("║          Bahasa Ocu → Bahasa Indonesia                      ║")
    print("║          Menggunakan RAG + Ollama (Gemma)                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def interactive_mode(translator):
    """Menjalankan mode penerjemahan interaktif."""
    print("\nMode Interaktif - Ketik 'keluar' untuk mengakhiri")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🔤 Masukkan teks Bahasa Ocu: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['keluar', 'exit', 'quit', 'q']:
                print("\nTerima kasih telah menggunakan sistem ini!")
                break
            
            # Terjemahkan teks.
            print("\n⏳ Menerjemahkan...")
            result = translator.translate(user_input)
            
            if result.success:
                print(f"\n📝 Terjemahan:")
                print(f"   {result.translated_text}")
            else:
                print(f"\n❌ Error: {result.error_message}")
                
        except KeyboardInterrupt:
            print("\n\nTerima kasih!")
            break


def single_translation(translator, text: str):
    """Menerjemahkan satu teks."""
    result = translator.translate(text)
    
    if result.success:
        print(f"Input: {text}")
        print(f"Terjemahan: {result.translated_text}")
    else:
        print(f"Error: {result.error_message}")
        sys.exit(1)


def main():
    """Titik masuk utama aplikasi."""
    parser = argparse.ArgumentParser(
        description="Sistem Penerjemahan Bahasa Ocu ke Bahasa Indonesia"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Teks Bahasa Ocu yang akan diterjemahkan (opsional)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector index dari dataset"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Jalankan mode interaktif"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        # Buat penerjemah.
        print("Memuat sistem...")
        translator = create_translator(rebuild_index=args.rebuild)
        
        if args.text:
            # Mode penerjemahan tunggal.
            single_translation(translator, args.text)
        else:
            # Mode interaktif.
            interactive_mode(translator)
            
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Langkah-langkah setup:")
        print("   1. Salin file .env.example menjadi .env")
        print("   2. Pastikan Ollama aktif dan model Gemma sudah tersedia")
        print("   3. Set OLLAMA_BASE_URL/OLLAMA_GENERATE_ENDPOINT jika perlu")
        print("   4. Jalankan ulang program")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error tidak terduga: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
