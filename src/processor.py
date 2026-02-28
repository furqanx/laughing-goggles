import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

def get_processor(config):
    """
    Memuat Processor (FeatureExtractor + Tokenizer) menggunakan vocab.json offline.
    """
    save_dir = os.path.join(config['experiment']['output_dir'], "processor")
    
    # 1. Jika processor sudah pernah dibangun dan disimpan, load langsung (Fast Track)
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "vocab.json")):
        print(f"[Processor] Memuat processor utuh yang sudah ada dari {save_dir}")
        processor = Wav2Vec2Processor.from_pretrained(save_dir)
        return processor

    print("[Processor] Membangun Processor baru menggunakan vocab.json offline...")
    
    # 2. Inisialisasi Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=config['data']['sample_rate'], 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )

    # 3. Inisialisasi Tokenizer
    # Ambil path vocab.json yang sudah kita buat dari config
    vocab_path = config['data']['vocab_path'] 
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"❌ File vocab tidak ditemukan di: {vocab_path}. Jalankan notebook preprocessing terlebih dahulu!")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",   
        eos_token="</s>",  
        word_delimiter_token="|" 
    )
    
    # 4. Gabungkan dan Simpan
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    print(f"[Processor] ✅ Pembuatan processor selesai dan diamankan di {save_dir}")
    
    return processor