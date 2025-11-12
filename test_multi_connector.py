"""
Test script to verify multi-path connector implementation with dynamic token length
"""
import torch
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from models import ConnectorForMobileViT, ConnectorForBiLSTM
from transformers import AutoTokenizer

print("="*80)
print("MULTI-PATH CONNECTOR TEST with Dynamic Token Length")
print("="*80)

# Test parameters
batch_size = 3
mvit1_channels = 64
mvit2_channels = 128
bilstm_dim = 512
output_dim = 768  # GPT-2 hidden size

# Different widths for each path (simulating different MobileViT resolutions)
mvit1_width = 64  # W/8
mvit2_width = 32  # W/16
bilstm_width = 128  # Full width

# Test transcriptions with different lengths
transcriptions = [
    "Hello",  # Short (2 tokens)
    "This is a test",  # Medium (5 tokens)
    "This is a longer transcription with many words"  # Long (10 tokens)
]

print("\nTest transcriptions:")
for i, text in enumerate(transcriptions):
    print(f"{i+1}. \"{text}\"")

# Load tokenizer
print("\nLoading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Tokenize to get target lengths
print("\nTokenizing...")
tokenized = tokenizer(
    transcriptions,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

labels = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]
target_lengths = attention_mask.sum(dim=1)  # Actual token count per sample

print(f"\nTarget lengths per sample:")
for i in range(batch_size):
    print(f"  Sample {i+1}: {target_lengths[i].item()} tokens")
print(f"Max length in batch: {target_lengths.max().item()}")

# Create connectors
print("\n" + "="*80)
print("Creating Connectors...")
print("="*80)

connector_mvit1 = ConnectorForMobileViT(input_channels=mvit1_channels, output_dim=output_dim)
connector_mvit2 = ConnectorForMobileViT(input_channels=mvit2_channels, output_dim=output_dim)
connector_bilstm = ConnectorForBiLSTM(input_dim=bilstm_dim, output_dim=output_dim)

print(f"✓ ConnectorForMobileViT1: {mvit1_channels} → {output_dim}")
print(f"✓ ConnectorForMobileViT2: {mvit2_channels} → {output_dim}")
print(f"✓ ConnectorForBiLSTM: {bilstm_dim} → {output_dim}")

# Create dummy inputs with different widths
print("\n" + "="*80)
print("Creating dummy inputs...")
print("="*80)

# MobileViT1: (batch, 64, H, W)
mvit1_input = torch.randn(batch_size, mvit1_channels, 8, mvit1_width)
print(f"MobileViT1 input: {mvit1_input.shape} (batch, {mvit1_channels}, H, {mvit1_width})")

# MobileViT2: (batch, 128, H, W)
mvit2_input = torch.randn(batch_size, mvit2_channels, 4, mvit2_width)
print(f"MobileViT2 input: {mvit2_input.shape} (batch, {mvit2_channels}, H, {mvit2_width})")

# BiLSTM: (batch, seq_len, 512)
bilstm_input = torch.randn(batch_size, bilstm_width, bilstm_dim)
print(f"BiLSTM input: {bilstm_input.shape} (batch, {bilstm_width}, {bilstm_dim})")

# Test connectors WITHOUT target_lengths (original behavior)
print("\n" + "="*80)
print("TEST 1: Without target_lengths (original behavior)")
print("="*80)

output_mvit1_orig = connector_mvit1(mvit1_input)
output_mvit2_orig = connector_mvit2(mvit2_input)
output_bilstm_orig = connector_bilstm(bilstm_input)

print(f"MobileViT1 output: {output_mvit1_orig.shape} (expected: (3, {mvit1_width}, 768))")
print(f"MobileViT2 output: {output_mvit2_orig.shape} (expected: (3, {mvit2_width}, 768))")
print(f"BiLSTM output: {output_bilstm_orig.shape} (expected: (3, {bilstm_width}, 768))")

# Test connectors WITH target_lengths (new behavior)
print("\n" + "="*80)
print("TEST 2: With target_lengths (dynamic token length matching)")
print("="*80)

output_mvit1 = connector_mvit1(mvit1_input, target_lengths=target_lengths)
output_mvit2 = connector_mvit2(mvit2_input, target_lengths=target_lengths)
output_bilstm = connector_bilstm(bilstm_input, target_lengths=target_lengths)

max_len = target_lengths.max().item()
print(f"MobileViT1 output: {output_mvit1.shape} (expected: (3, {max_len}, 768))")
print(f"MobileViT2 output: {output_mvit2.shape} (expected: (3, {max_len}, 768))")
print(f"BiLSTM output: {output_bilstm.shape} (expected: (3, {max_len}, 768))")

# Verify shapes match
print("\n" + "="*80)
print("SHAPE VERIFICATION:")
print("="*80)

all_match = (
    output_mvit1.shape == output_mvit2.shape == output_bilstm.shape and
    output_mvit1.shape[0] == batch_size and
    output_mvit1.shape[1] == max_len and
    output_mvit1.shape[2] == output_dim
)

if all_match:
    print("✅ All connector outputs have matching shapes!")
    print(f"   Shape: ({batch_size}, {max_len}, {output_dim})")
else:
    print("❌ Shape mismatch detected!")
    sys.exit(1)

# Verify labels and attention_mask match
if labels.shape[1] == max_len and attention_mask.shape[1] == max_len:
    print(f"✅ Labels and attention_mask match connector output length!")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
else:
    print("❌ Labels/attention_mask length mismatch!")
    sys.exit(1)

# Test with different batch (different max length)
print("\n" + "="*80)
print("TEST 3: Different batch with different max length")
print("="*80)

transcriptions2 = ["Short", "Longer text here"]
tokenized2 = tokenizer(
    transcriptions2,
    return_tensors="pt",
    padding=True,
    truncation=True
)
target_lengths2 = tokenized2["attention_mask"].sum(dim=1)
max_len2 = target_lengths2.max().item()

print(f"New target lengths: {target_lengths2.tolist()}")
print(f"New max length: {max_len2}")

batch_size2 = len(transcriptions2)
mvit1_input2 = torch.randn(batch_size2, mvit1_channels, 8, mvit1_width)
output_mvit1_2 = connector_mvit1(mvit1_input2, target_lengths=target_lengths2)

if output_mvit1_2.shape[1] == max_len2:
    print(f"✅ Dynamic length adjustment working: {max_len} → {max_len2}")
    print(f"   Output shape: {output_mvit1_2.shape}")
else:
    print("❌ Dynamic length adjustment failed!")
    sys.exit(1)

# Test attention_mask masking
print("\n" + "="*80)
print("TEST 4: Attention mask verification")
print("="*80)

print("Attention mask:")
for i in range(batch_size):
    mask_str = ''.join(['1' if m == 1 else '0' for m in attention_mask[i].tolist()])
    print(f"  Sample {i+1}: {mask_str} ({target_lengths[i].item()} real tokens)")

# Verify padding positions
for i in range(batch_size):
    real_tokens = target_lengths[i].item()
    if attention_mask[i, :real_tokens].sum() == real_tokens and \
       attention_mask[i, real_tokens:].sum() == 0:
        print(f"✅ Sample {i+1}: Correct masking")
    else:
        print(f"❌ Sample {i+1}: Incorrect masking!")
        sys.exit(1)

# Summary
print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nSummary:")
print("- 3 connectors (MobileViT1, MobileViT2, BiLSTM) working correctly")
print("- Dynamic token length adjustment working for all paths")
print("- All connector outputs match target token lengths")
print("- Attention masks correctly identify real vs padding tokens")
print("- Different batches can have different max lengths")
print("\nImplementation ready for training!")
