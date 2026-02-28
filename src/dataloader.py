import os
import random
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from src.preprocessing import DynamicNoiseInjector

# ==========================================
# 1. DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pisahkan input_values (audio) dan labels (teks fonem)
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Padding Audio menggunakan Feature Extractor
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # REVISI 2: Padding Label (Token ID) secara eksplisit menggunakan Tokenizer
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        # Ganti padding token di labels menjadi -100 agar diabaikan oleh PyTorch Loss (CTC)
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        batch["labels"] = labels
        return batch

# ==========================================
# 2. BUCKET BATCH SAMPLER
# ==========================================
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Cari tahu nama kolom durasi yang benar (antisipasi perbedaan penamaan)
        df_cols = dataset.data.columns
        self.dur_col = 'audio_duration_sec' if 'audio_duration_sec' in df_cols else 'duration'

        self.ind_n_len = []
        for i in range(len(dataset)):
            self.ind_n_len.append((i, dataset.data.iloc[i].get(self.dur_col, 0)))
        
        # Urutkan berdasarkan durasi (terpendek ke terpanjang)
        self.ind_n_len.sort(key=lambda x: x[1])
        self.batches = [self.ind_n_len[i:i + batch_size] for i in range(0, len(self.ind_n_len), batch_size)]
        
    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield [idx for idx, _ in batch]
            
    def __len__(self):
        return len(self.batches)

# ==========================================
# 3. DATASET KUSTOM
# ==========================================
class ASRDataset(Dataset):
    def __init__(self, data, processor, target_sr=16000, augmentor=None, target_col='phonetic_text'):
        self.data = data
        self.processor = processor
        self.target_sr = target_sr 
        self.augmentor = augmentor
        self.target_col = target_col 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        audio_path = row['audio_path']
        transcript = row[self.target_col] 

        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sample_rate != self.target_sr:
            # REVISI 3: Gunakan fungsi fungsional agar tidak memberatkan CPU di setiap iterasi
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.target_sr)

        waveform = waveform.squeeze(0).numpy()

        if self.augmentor is not None:
            waveform = self.augmentor(waveform_np=waveform, sample_rate=self.target_sr)

        input_values = self.processor(waveform, sampling_rate=self.target_sr).input_values[0]
        
        # REVISI 1: Gunakan pemanggilan tokenizer modern (as_target_processor sudah deprecated)
        labels = self.processor.tokenizer(transcript).input_ids

        return {
            "input_values": input_values,
            "labels": labels
        }

def filter_data(df, min_duration=0.5, max_duration=15.0):
    # REVISI 4: Pengecekan kolom durasi dinamis
    dur_col = 'audio_duration_sec' if 'audio_duration_sec' in df.columns else 'duration'
    
    if dur_col in df.columns:
        return df[(df[dur_col] >= min_duration) & (df[dur_col] <= max_duration)].reset_index(drop=True)
    return df

# ==========================================
# 4. GET DATALOADER
# ==========================================
def get_dataloader(config, processor):
    print(f"[DataLoader] Membaca manifest: {config['train_manifest']}")
    
    train_df = pd.read_json(config['train_manifest'], lines=True)
    val_df = pd.read_json(config['val_manifest'], lines=True)
    
    min_dur = config.get('min_duration', 0.5)
    max_dur = config.get('max_duration', 15.0)
    target_col = config.get('target_column', 'text') 

    print(f"[DataLoader] Filter durasi: {min_dur}s - {max_dur}s")
    train_df = filter_data(train_df, min_duration=min_dur, max_duration=max_dur)
    val_df = filter_data(val_df, min_duration=min_dur, max_duration=20.0)
    print(f"[DataLoader] Sisa data setelah difilter -> Train: {len(train_df)}, Val: {len(val_df)}")

    target_sr = config.get('sample_rate', 16000)
    
    noise_dir = config.get('noise_dir', './data/raw/noise')
    noise_injector = None
    if os.path.exists(noise_dir):
        noise_prob = config.get('noise_prob', 0.5) 
        noise_injector = DynamicNoiseInjector(noise_dir=noise_dir, p=noise_prob)
        print(f"[DataLoader] Augmentasi Noise diaktifkan (Prob: {noise_prob}) dari {noise_dir}")
    else:
        print(f"[DataLoader] ⚠️ Peringatan: Folder noise '{noise_dir}' tidak ditemukan. Augmentasi dinonaktifkan.")

    train_ds = ASRDataset(
        data=train_df, processor=processor, target_sr=target_sr, 
        augmentor=noise_injector, target_col=target_col
    )
    val_ds = ASRDataset(
        data=val_df, processor=processor, target_sr=target_sr, 
        augmentor=None, target_col=target_col
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    train_sampler = BucketBatchSampler(train_ds, batch_size=config['batch_size'])

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=config.get('num_workers', 4), 
        collate_fn=data_collator, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=data_collator,
        pin_memory=True
    )
    
    return train_loader, val_loader