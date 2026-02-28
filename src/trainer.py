import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
from src.utils.metrics import compute_asr_metrics 

class ASRTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, processor, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.processor = processor
        self.logger = logger 
        
        self.output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        self.accum_steps = config['train'].get('gradient_accumulation_steps', 1)
        self.global_step = 0 

        # ==========================================
        # LOSS FUNCTIONS
        # ==========================================
        self.pad_id = self.processor.tokenizer.pad_token_id
        self.ctc_loss_fn = nn.CTCLoss(blank=self.pad_id, reduction='mean', zero_infinity=True)
        self.attn_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
        self.alpha = config['train'].get('ctc_weight', 1.0) 

    def train(self):
        epochs = self.config['train']['epochs']
        best_per = float('inf') 
        
        print("\n[Trainer] Memulai Siklus Pelatihan...")
        for epoch in range(epochs):
            # 1. Jalankan Training
            train_loss = self._train_epoch(epoch)
            
            # 2. Jalankan Validasi
            print(f"\nEpoch {epoch+1} Selesai. Mengevaluasi Model...")
            val_metrics = self.validate()
            
            # Amankan metrik (jaga-jaga jika fungsi metrik mengembalikan nama key yang beda)
            current_per = val_metrics.get('per', val_metrics.get('cer', 1.0))
            
            if self.logger:
                self.logger.log({
                    "epoch/train_loss": train_loss,
                    "epoch/val_per": current_per,
                    "epoch": epoch + 1
                }, step=self.global_step)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val PER: {current_per:.4f}")
            
            # 3. Simpan Checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            if current_per < best_per:
                print(f"🔥 Rekor Baru! Menyimpan Model Terbaik (PER: {best_per:.4f} -> {current_per:.4f})")
                best_per = current_per
                self.save_checkpoint(epoch, val_metrics, is_best=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for step, batch in enumerate(pbar):
            input_values = batch['input_values'].to(self.device)
            # REVISI 1: Ekstrak attention_mask yang dibuat oleh collator
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            labels = batch['labels'].to(self.device)
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=(self.scaler is not None)):
                # Oper attention_mask ke model
                outputs = self.model(
                    input_values=input_values, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                ctc_logits = outputs.get("ctc_logits")       
                decoder_logits = outputs.get("decoder_logits") 
                input_lengths = outputs.get("input_lengths") 

                # --- 1. CTC LOSS ---
                # PyTorch CTCLoss minta logits berbentuk (Waktu, Batch, Vocab)
                ctc_logits_t = ctc_logits.transpose(0, 1).log_softmax(2)
                
                # REVISI 2: Cara menghitung target yang jauh lebih aman dan direkomendasikan PyTorch
                target_lengths = torch.sum(labels != -100, dim=1).long()
                # Ganti -100 dengan pad_id agar tidak error saat masuk CTCLoss 
                # (CTCLoss tidak mengerti ignore_index=-100, ia hanya mengerti argument 'blank')
                safe_targets = torch.where(labels == -100, self.pad_id, labels)
                
                loss_ctc = self.ctc_loss_fn(ctc_logits_t, safe_targets, input_lengths, target_lengths)

                # --- 2. HYBRID / ATTENTION LOSS ---
                if decoder_logits is not None and self.alpha < 1.0:
                    vocab_size = decoder_logits.size(-1)
                    flat_decoder_logits = decoder_logits.reshape(-1, vocab_size)
                    flat_labels = labels.reshape(-1)
                    loss_attn = self.attn_loss_fn(flat_decoder_logits, flat_labels)
                    loss = (self.alpha * loss_ctc) + ((1 - self.alpha) * loss_attn)
                else:
                    loss = loss_ctc 

                loss = loss / self.accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.logger:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                    self.logger.log({
                        "train/step_loss": loss.item() * self.accum_steps,
                        "train/learning_rate": current_lr
                    }, step=self.global_step)

            total_loss += loss.item() * self.accum_steps
            pbar.set_postfix({'loss': loss.item() * self.accum_steps})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_values=input_values, 
                    attention_mask=attention_mask
                )
                
                ctc_logits = outputs["ctc_logits"]
                pred_ids = torch.argmax(ctc_logits, dim=-1)
                
                labels = torch.where(labels == -100, self.pad_id, labels)
                
                # PERINGATAN: batch_decode dengan group_tokens=True adalah kunci untuk menghapus duplikasi CTC
                pred_str = self.processor.batch_decode(pred_ids, group_tokens=True)
                # Untuk label asli, JANGAN di-group agar tidak merusak huruf ganda yang valid
                label_str = self.processor.batch_decode(labels, group_tokens=False)
                
                all_preds.extend(pred_str)
                all_labels.extend(label_str)

        metrics = compute_asr_metrics(all_preds, all_labels)
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        filename = "checkpoint_best.pth" if is_best else "checkpoint_last.pth"
        save_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, save_path)
        
        if is_best:
            self.processor.save_pretrained(os.path.join(self.output_dir, "best_processor"))