import torch
import torch.nn as nn
from transformers import AutoModelForCTC, AutoConfig

class EndToEndCTCASR(nn.Module):
    def __init__(self, vocab_size, pad_token_id, config):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        backbone_name = config.get("backbone", "microsoft/wavlm-base-plus")
        print(f"[Model Factory] Merakit Model Murni CTC dengan Backbone: {backbone_name}")
        
        hf_config = AutoConfig.from_pretrained(
            backbone_name,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            ctc_loss_reduction="mean",
            mask_time_prob=config.get("mask_time_prob", 0.05), 
            mask_time_length=config.get("mask_time_length", 10),
            mask_feature_prob=config.get("mask_feature_prob", 0.05),
            mask_feature_length=config.get("mask_feature_length", 10)
        )
        
        self.backbone = AutoModelForCTC.from_pretrained(
            backbone_name, 
            config=hf_config,
            ignore_mismatched_sizes=True 
        )
        
        # REVISI 2: Gunakan nama method yang baru agar terhindar dari deprecation error
        if config.get("freeze_feature_extractor", True):
            print("[Model Factory] Membekukan CNN Feature Encoder (Sangat Disarankan)")
            if hasattr(self.backbone, "freeze_feature_encoder"):
                self.backbone.freeze_feature_encoder()
            else:
                # Fallback untuk versi transformers lama
                self.backbone.freeze_feature_extractor()

    # REVISI 1: Tambahkan attention_mask dan target_tokens opsional (labels) ke arguments
    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        """
        input_values: Audio gelombang mentah (Batch Size, Panjang Audio)
        attention_mask: Masker boolean untuk membedakan audio asli dan padding
        """
        
        # Teruskan attention_mask ke otak Transformer agar padding diabaikan
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        ctc_logits = outputs.logits # Shape: (Batch, Waktu, Vocab)
        
        # REVISI 3: Kalkulasi panjang fitur yang lebih elegan dan akurat menggunakan mask
        if attention_mask is not None:
            # Hitung durasi asli setiap sampel di batch sebelum sub-sampling CNN
            input_lengths_raw = attention_mask.sum(-1).to(torch.long)
        else:
            batch_size = input_values.size(0)
            input_lengths_raw = torch.full((batch_size,), input_values.size(1), dtype=torch.long, device=input_values.device)
            
        # Hitung panjang output setelah melewati lapisan CNN (untuk CTCLoss)
        input_lengths = self.backbone._get_feat_extract_output_lengths(input_lengths_raw)
        
        return {
            "ctc_logits": ctc_logits, 
            "decoder_logits": None,  
            "input_lengths": input_lengths
        }

# ==========================================
# FACTORY FUNCTION
# ==========================================
def build_model(config, processor):
    vocab_size = len(processor.tokenizer)
    pad_token_id = processor.tokenizer.pad_token_id
    
    model = EndToEndCTCASR(vocab_size=vocab_size, pad_token_id=pad_token_id, config=config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Senjata (Trainable Parameters): {trainable_params:,}")
    
    return model