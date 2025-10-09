import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Wav2Vec2ConformerForCTC
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig


# Check if CUDA is available
device_llm = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Load the LLaMA model and tokenizer]
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = "right"

# Configure the model for 4-bit quantization
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16
}

# Load the LLM explicitly on device 1
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_cache=False
).to(device_llm)

for param in llm_model.parameters():
    param.requires_grad = False

llm_hidden_size = llm_model.config.hidden_size