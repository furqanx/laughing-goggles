# 🎯 Child Speech-to-Phoneme ASR: End-to-End WavLM CTC

Repositori ini berisi *pipeline State-of-the-Art* (SOTA) Automatic Speech Recognition (ASR) yang dirancang khusus untuk mengekstraksi dan mengenali transkripsi fonem (IPA) dari suara anak-anak. Sistem ini dibangun untuk menangani tantangan akustik tingkat tinggi seperti durasi ucapan yang fluktuatif, *noise* latar belakang, dan artikulasi fonetik yang masih berkembang.

Dibangun dengan fokus pada efisiensi *audio signal processing*, arsitektur ini menggunakan pendekatan **End-to-End Transformer Finetuning dengan Pure CTC Loss**:

1. **Offline Phoneme Normalization**: Teks target divalidasi dan dinormalisasi secara *offline* menjadi 52 karakter baku IPA (International Phonetic Alphabet). Menghasilkan `vocab.json` deterministik untuk memastikan sinkronisasi absolut antara prediksi model dan metrik penilai (Kaggle).
2. **On-the-Fly Audio Processing**: Memuat raw audio secara efisien menggunakan `torchaudio`, dilengkapi dengan *dynamic resampling* dan *dynamic noise injection* langsung di level `DataLoader`. Didukung oleh `BucketBatchSampler` untuk meminimalisir *padding waste*.
3. **End-to-End WavLM Finetuning**: Menggunakan *backbone* `microsoft/wavlm-base-plus`. CNN Feature Encoder dibekukan (frozen) untuk mempertahankan ekstraksi representasi sinyal dasar yang kokoh, sementara lapisan Transformer dibiarkan berlatih menggunakan fungsi *Connectionist Temporal Classification* (CTC) *Loss* beserta *attention mask* dinamis.

## 📁 Struktur Markas (Repository Tree)

```text
.
├── data/
│   ├── raw/                      # Audio mentah (.flac/.wav) dan folder noise augmentasi
│   ├── interim/                  # Manifest transkrip awal yang belum dinormalisasi
│   └── processed/                # Manifest bersih (train/val.jsonl) & vocab.json (52 IPA chars)
├── logs/
│   └── experiments/              # Checkpoint model terbaik, custom processor, & config backup
├── src/
│   ├── utils/
│   │   ├── logger.py             # Integrasi Weights & Biases (W&B) untuk tracking eksperimen
│   │   └── metrics.py            # Kalkulasi Phoneme Error Rate (PER) di level karakter via JiWER
│   ├── preprocessing.py          # Modul injeksi noise dinamis untuk augmentasi audio
│   ├── dataloader.py             # PyTorch Dataloader + BucketBatchSampler + CTC Padding Collator
│   ├── model.py                  # End-to-End CTC Wrapper untuk HuggingFace AutoModelForCTC
│   ├── processor.py              # Handler Feature Extractor & Tokenizer menggunakan vocab offline
│   └── trainer.py                # Loop training kustom dengan AMP (FP16/BF16) & Checkpointing
├── base_config.yaml              # Pusat komando (Hyperparameters, Model Backbone, Paths)
└── train.py                      # Skrip utama orkestrasi & eksekusi eksperimen

```