# model_llm.py
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer


# --- 1) Loader ---
def build_llama(model_name: str, dtype: str = "bfloat16", device: str = "cuda"):
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    llama = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=dt).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return llama, tok

# --- 2) Conditioner ---
class PrefixConditionerLite(nn.Module):
    """
    (B,Tp,d_llm) の prefix を LLaMA の埋め込み列の先頭へ前置。
    labels には prefix 長ぶん -100 を前置して損失から除外。
    """
    def __init__(self, llama: LlamaForCausalLM, d_llm: int):
        super().__init__()
        self.llama = llama
        H = llama.config.hidden_size
        self.align = nn.Identity() if d_llm == H else nn.Linear(d_llm, H, bias=False)

    @property
    def device(self): return next(self.llama.parameters()).device
    @property
    def dtype(self):  return self.llama.model.embed_tokens.weight.dtype

    def forward(self, llm_prefix, input_ids=None, labels=None, attention_mask=None):
        assert llm_prefix is not None, "llm_prefix is None"
        assert llm_prefix.dim() == 3, f"llm_prefix must be (B,Tp,d), got {llm_prefix.shape}"
        B, Tp, _ = llm_prefix.shape

        prefix = self.align(llm_prefix.to(self.device)).to(self.dtype)  # (B,Tp,H)

        if input_ids is not None:
            tok_emb = self.llama.model.embed_tokens(input_ids.to(self.device)).to(self.dtype)  # (B,Tt,H)
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)                                # (B,Tp+Tt,H)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
            attn = torch.cat([torch.ones(B, Tp, device=self.device, dtype=attention_mask.dtype),
                              attention_mask], dim=1)
        else:
            inputs_embeds = prefix
            attn = torch.ones(B, Tp, device=self.device, dtype=torch.long)

        lm_labels = None
        if labels is not None:
            ignore = torch.full((B, Tp), -100, dtype=torch.long, device=self.device)
            lm_labels = torch.cat([ignore, labels.to(self.device)], dim=1)

        return self.llama(inputs_embeds=inputs_embeds,
                          attention_mask=attn,
                          labels=lm_labels,
                          use_cache=True,
                          return_dict=True)

# --- 3) lalm 損失（Language-Aware LM Loss） ---
class LALMLoss(nn.Module):
    """
    texts（生文字列）と llm_prefix から CE loss を返す。
    freeze_llama=True なら LLaMA を凍結して視覚側（Connector/CRNN）にだけ勾配を返す。
    detach_prefix=True なら視覚側にも勾配を返さず LLM 側だけ最適化（通常は False 推奨）。
    """
    def __init__(self, conditioner: PrefixConditionerLite, tokenizer, 
                 freeze_llama: bool = True, detach_prefix: bool = False):
        super().__init__()
        self.cond = conditioner
        self.tok = tokenizer
        self.detach_prefix = detach_prefix
        if freeze_llama:
            self.cond.llama.requires_grad_(False).eval()

    def forward(self, llm_prefix, texts):
        if llm_prefix is None:
            raise ValueError("llm_prefix is None. Ensure HTR head produced it in training.")
        if self.detach_prefix:
            llm_prefix = llm_prefix.detach()
        enc = self.tok(list(texts), padding=True, return_tensors="pt").to(self.cond.device)
        out = self.cond(llm_prefix, input_ids=enc.input_ids, labels=enc.input_ids, attention_mask=enc.attention_mask)
        return out.loss
