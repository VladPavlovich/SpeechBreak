import sys
from transformers import AutoProcessor, Qwen2AudioConfig, Qwen2AudioForConditionalGeneration
out = sys.argv[1]
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
cfg = Qwen2AudioConfig(
    audio_config=dict(d_model=64, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=128, num_mel_bins=128),
    text_config=dict(model_type="qwen2", hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
                     num_key_value_heads=2, intermediate_size=128, vocab_size=len(proc.tokenizer)),
    audio_token_index=proc.audio_token_id,
)
import torch; torch.manual_seed(0)
Qwen2AudioForConditionalGeneration(cfg).save_pretrained(out); proc.save_pretrained(out)
print("saved tiny model; vocab", len(proc.tokenizer), "audio_token_id", proc.audio_token_id)
