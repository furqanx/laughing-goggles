import numpy as np
import re

try:
    import jiwer
except ImportError:
    print("[Metrics] Warning: Library 'jiwer' belum terinstall. Metrik PER akan gagal.")
    jiwer = None

def compute_asr_metrics(pred_strs, label_strs):
    """
    Menghitung Phoneme Error Rate (PER) menggunakan metrik level karakter.
    PER secara matematis ekuivalen dengan CER saat menghitung jarak Levenshtein antar fonem IPA.
    """
    if jiwer is None:
        return {"per": 1.0}

    clean_preds = []
    clean_labels = []

    for p, l in zip(pred_strs, label_strs):
        # 1. Hancurkan token spesial yang "bocor" dari Tokenizer
        tokens_to_remove = ['[PAD]', '[UNK]', '<s>', '</s>', '<CTC_BLANK>']
        for t in tokens_to_remove:
            p = p.replace(t, '')
            l = l.replace(t, '')

        # 2. REVISI KRUSIAL: Penanganan Delimiter Batas Kata
        # "|" adalah representasi model untuk spasi " ". 
        # Kita WAJIB mengembalikannya menjadi spasi agar dihitung benar oleh jiwer.
        p = p.replace('|', ' ')
        l = l.replace('|', ' ')

        # 3. Normalisasi Spasi (TANPA .lower() untuk menjaga integritas IPA)
        # Menghapus spasi ganda dan spasi di ujung string
        p = re.sub(r'\s+', ' ', p).strip()
        l = re.sub(r'\s+', ' ', l).strip()

        clean_preds.append(p)
        clean_labels.append(l)

    # 4. Filter Penyelamat: jiwer akan crash jika Ground Truth (label) benar-benar kosong ""
    valid_preds = []
    valid_labels = []
    for p, l in zip(clean_preds, clean_labels):
        if len(l) > 0: 
            valid_preds.append(p)
            valid_labels.append(l)
            
    if len(valid_labels) == 0:
         return {"per": 0.0}

    # 5. Eksekusi Perhitungan PER (CER pada level karakter IPA)
    per_score = jiwer.cer(valid_labels, valid_preds)

    return {
        "per": per_score
    }