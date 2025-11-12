"""
Verification script to check if only connector layers are trainable
"""
import torch
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from models import HTRNet
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load('config.yaml')

# Create model
use_llm = config.train.get('use_llm', True)
nclasses = 100  # dummy value
net = HTRNet(config.arch, nclasses, use_llm=use_llm)

# Freeze all except connectors
net.freeze_except_connectors()

print("\n" + "="*80)
print("PARAMETER FREEZE VERIFICATION")
print("="*80 + "\n")

# Check which parameters are trainable
trainable_components = {}
frozen_components = {}

for name, param in net.named_parameters():
    component = name.split('.')[0]  # Get top-level component

    if param.requires_grad:
        if component not in trainable_components:
            trainable_components[component] = []
        trainable_components[component].append(name)
    else:
        if component not in frozen_components:
            frozen_components[component] = []
        frozen_components[component].append(name)

print("TRAINABLE COMPONENTS:")
print("-" * 80)
for component, params in trainable_components.items():
    param_count = sum(p.numel() for n, p in net.named_parameters() if n in params)
    print(f"\n{component}: {len(params)} parameters ({param_count:,} values)")
    # Show first 3 parameter names as examples
    for param_name in params[:3]:
        print(f"  - {param_name}")
    if len(params) > 3:
        print(f"  ... and {len(params)-3} more")

print("\n" + "="*80)
print("FROZEN COMPONENTS:")
print("-" * 80)
for component, params in frozen_components.items():
    param_count = sum(p.numel() for n, p in net.named_parameters() if n in params)
    print(f"\n{component}: {len(params)} parameters ({param_count:,} values)")

# Detailed check for connectors in CTCtopB
print("\n" + "="*80)
print("CONNECTOR LAYER DETAILS:")
print("-" * 80)
if hasattr(net.top, 'connector_mvit1'):
    for name, param in net.top.connector_mvit1.named_parameters():
        status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
        print(f"{status}: connector_mvit1.{name} - {param.numel():,} values")

if hasattr(net.top, 'connector_mvit2'):
    for name, param in net.top.connector_mvit2.named_parameters():
        status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
        print(f"{status}: connector_mvit2.{name} - {param.numel():,} values")

if hasattr(net.top, 'connector_bilstm'):
    for name, param in net.top.connector_bilstm.named_parameters():
        status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
        print(f"{status}: connector_bilstm.{name} - {param.numel():,} values")

# Summary
print("\n" + "="*80)
print("SUMMARY:")
print("-" * 80)
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"Total parameters:      {total_params:,}")
print(f"Trainable parameters:  {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Frozen parameters:     {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
print("="*80 + "\n")

# Verify expectations
success = True
if 'top' not in trainable_components or \
   not any('connector' in param for params in trainable_components.values() for param in params):
    print("❌ ERROR: Connector layers are not trainable!")
    success = False
else:
    print("✓ Connector layers are trainable")

if 'features' in trainable_components:
    print("❌ ERROR: CNN backbone (features) should be frozen!")
    success = False
else:
    print("✓ CNN backbone is frozen")

if success:
    print("\n✅ VERIFICATION PASSED: Only connector layers are trainable!")
else:
    print("\n❌ VERIFICATION FAILED: Check the freeze implementation!")
    sys.exit(1)
