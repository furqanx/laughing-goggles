import os
import wandb
from typing import Dict, Any

class WandbLogger:
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Wrapper untuk Weights & Biases agar tidak mengotori train.py.
        """
        self.enabled = enabled
        
        # Cek apakah user ingin menggunakan wandb dari config
        # Jika false, logger akan berjalan dalam mode "dummy" (tidak melakukan apa-apa)
        if self.enabled:
            # Mengambil informasi dari config yaml Anda
            exp_config = config.get('experiment', {})
            # Menetapkan default name yang relevan dengan task kita jika tidak ada di config
            project_name = exp_config.get('project_name', 'children-speech-to-phoneme')
            run_name = exp_config.get('run_name', 'baseline-run')
            
            print(f"[Logger] Menginisialisasi Weights & Biases. Project: {project_name}")
            
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config, # Otomatis melacak semua hyperparameter dari yaml
                reinit=True
            )
        else:
            print("[Logger] Weights & Biases DIMATIKAN (Dummy Mode).")

    def log(self, metrics: Dict[str, float], step: int = None):
        """
        Digunakan di dalam training loop (src/trainer.py) untuk mencatat loss dan PER.
        Contoh: logger.log({"train_loss": 0.5, "val_per": 0.12}, step=100)
        """
        if self.enabled:
            wandb.log(metrics, step=step)

    def watch_model(self, model):
        """
        (Opsional) Melacak distribusi bobot dan gradien model untuk mendeteksi vanishing/exploding gradients.
        """
        if self.enabled:
            wandb.watch(model, log="all", log_freq=100)

    def finish(self):
        """
        Menutup sesi wandb dengan aman setelah training selesai atau terinterupsi.
        """
        if self.enabled:
            wandb.finish()
            print("[Logger] Weights & Biases run ditutup.")