"""
Mobile ViTの真のアテンションマップ（Query-Key-Value Attention）を可視化

このスクリプトは、TransformerEncoderLayerのアテンション重みを直接取得します。
PyTorchのnn.TransformerEncoderLayerを一時的にラップして、アテンション重みを返すようにします。

使い方:
    python debug/visualize_true_attention.py

出力:
    - results/[timestamp]/true_attention_*.png: 真のアテンションマップ
    - results/[timestamp]/attention_head_*.png: 各アテンションヘッドの可視化
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from models import HTRNet
from utils.htr_dataset import HTRDataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 結果保存用ディレクトリ
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', f'true_attention_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Created results directory: {results_dir}\n")


class AttentionWrapper:
    """
    TransformerEncoderLayerのアテンション重みをキャプチャするラッパー
    """

    def __init__(self):
        self.attention_maps = []

    def wrap_transformer(self, transformer_encoder):
        """
        TransformerEncoderをラップして、アテンション重みを保存できるようにする

        Args:
            transformer_encoder: nn.TransformerEncoder
        """
        self.attention_maps = []

        # 各レイヤーのself_attnをラップ
        for layer_idx, layer in enumerate(transformer_encoder.layers):
            # 元のself_attn（MultiheadAttention）を保存
            original_self_attn = layer.self_attn

            # ラップされたforward関数を作成
            def create_wrapped_forward(original_attn, layer_id):
                original_forward = original_attn.forward

                def wrapped_forward(*args, **kwargs):
                    # need_weights=True, average_attn_weights=Falseを強制
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False

                    output, attn_weights = original_forward(*args, **kwargs)

                    # アテンション重みを保存
                    self.attention_maps.append({
                        'layer_idx': layer_id,
                        'attn_weights': attn_weights.detach().cpu(),  # (batch, num_heads, seq_len, seq_len)
                    })

                    return output, attn_weights

                return wrapped_forward

            # forward関数を置き換え
            layer.self_attn.forward = create_wrapped_forward(original_self_attn, layer_idx)

    def get_attention_maps(self):
        """保存されたアテンション重みを取得"""
        return self.attention_maps

    def clear(self):
        """保存されたアテンション重みをクリア"""
        self.attention_maps = []


def visualize_attention_maps(attention_maps, img, transcr, save_prefix, results_dir):
    """
    アテンションマップを可視化

    Args:
        attention_maps: List of dict with 'layer_idx' and 'attn_weights'
        img: 入力画像 (1, C, H, W)
        transcr: 正解文字列
        save_prefix: ファイル名のプレフィックス
        results_dir: 保存先ディレクトリ
    """

    if len(attention_maps) == 0:
        print("WARNING: No attention maps captured!")
        return

    num_layers = len(attention_maps)

    # レイヤーごとのアテンションマップを可視化
    for layer_data in attention_maps:
        layer_idx = layer_data['layer_idx']
        attn_weights = layer_data['attn_weights']  # (batch, num_heads, seq_len, seq_len)

        batch_size, num_heads, seq_len, _ = attn_weights.shape

        # 各ヘッドのアテンションマップを可視化
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(16, 6))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            attn_map = attn_weights[0, head_idx].numpy()  # (seq_len, seq_len)

            im = axes[head_idx].imshow(attn_map, cmap='viridis', aspect='auto')
            axes[head_idx].set_title(f'Head {head_idx + 1}', fontsize=10)
            axes[head_idx].set_xlabel('Key Position', fontsize=8)
            axes[head_idx].set_ylabel('Query Position', fontsize=8)
            plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)

        # 使用していない軸を非表示
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Layer {layer_idx + 1} Attention Heads\nGT: "{transcr}"', fontsize=14)
        plt.tight_layout()

        output_path = os.path.join(results_dir, f'{save_prefix}_layer{layer_idx + 1}_heads.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved attention heads: {output_path}")

        # 平均アテンションマップを可視化
        avg_attn = attn_weights[0].mean(dim=0).numpy()  # (seq_len, seq_len)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(avg_attn, cmap='hot', aspect='auto')
        ax.set_title(f'Layer {layer_idx + 1} - Average Attention (all heads)\nGT: "{transcr}"', fontsize=14)
        ax.set_xlabel('Key Position (Patch Index)', fontsize=12)
        ax.set_ylabel('Query Position (Patch Index)', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        output_path = os.path.join(results_dir, f'{save_prefix}_layer{layer_idx + 1}_avg.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved average attention: {output_path}")

    # すべてのレイヤーの平均アテンションを1つの図にまとめる
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    for idx, layer_data in enumerate(attention_maps):
        attn_weights = layer_data['attn_weights']
        avg_attn = attn_weights[0].mean(dim=0).numpy()  # (seq_len, seq_len)

        im = axes[idx].imshow(avg_attn, cmap='hot', aspect='auto')
        axes[idx].set_title(f'Layer {idx + 1}', fontsize=12)
        axes[idx].set_xlabel('Key Position', fontsize=10)
        axes[idx].set_ylabel('Query Position', fontsize=10)
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'Average Attention Maps (All Layers)\nGT: "{transcr}"', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(results_dir, f'{save_prefix}_all_layers.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved all layers summary: {output_path}")


# 設定ロード
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
config = OmegaConf.load(config_path)

device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# データセット準備
print("Loading dataset...")
data_path = config.data.path
if not os.path.isabs(data_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, data_path)

dataset = HTRDataset(
    data_path,
    'test',
    fixed_size=(config.preproc.image_height, config.preproc.image_width)
)

# 文字クラスのロード
classes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'saved_models', 'classes.npy')
classes = np.load(classes_path, allow_pickle=True).tolist()
print(f"Loaded {len(classes)} character classes\n")

# モデル作成
print("Creating model...")
net = HTRNet(config.arch, len(classes) + 1, use_llm=False)

# 学習済み重みのロード
if config.resume and os.path.exists(config.resume):
    print(f"Loading checkpoint: {config.resume}")
    load_dict = torch.load(config.resume, map_location='cpu')
    net.load_state_dict(load_dict, strict=False)
    print("Loaded checkpoint\n")

net.to(device)
net.eval()

# AttentionWrapperを作成
attention_wrapper = AttentionWrapper()

# モデル内のMobileViTブロックを探してラップ
print("Wrapping MobileViT Transformer layers...")
mvit_blocks = []
for name, module in net.named_modules():
    if hasattr(module, 'transformer') and isinstance(module.transformer, nn.TransformerEncoder):
        print(f"  Found MobileViT block: {name}")
        attention_wrapper.wrap_transformer(module.transformer)
        mvit_blocks.append(name)

if len(mvit_blocks) == 0:
    print("ERROR: No MobileViT blocks found! Check your model architecture.")
    sys.exit(1)

print(f"Wrapped {len(mvit_blocks)} MobileViT blocks\n")

# メイン処理
print("="*80)
print("Visualizing True Attention Maps (Query-Key-Value)")
print("="*80)

# サンプル選択
num_samples = 5
sample_indices = [5, 37, 12, 4, 67]

for i, idx in enumerate(sample_indices):
    img, transcr = dataset[idx]
    img = img.unsqueeze(0).to(device)

    print(f"\n{'='*80}")
    print(f"Sample {i+1}/{num_samples} (Index: {idx})")
    print(f"{'='*80}")
    print(f"Ground Truth: '{transcr}'")

    # アテンションマップをクリア
    attention_wrapper.clear()

    # モデルを実行してアテンション重みをキャプチャ
    with torch.no_grad():
        _ = net(img)

    # アテンションマップを取得
    attention_maps = attention_wrapper.get_attention_maps()
    print(f"   Captured {len(attention_maps)} attention maps")

    # 可視化
    visualize_attention_maps(attention_maps, img, transcr, f'sample_{i+1}', results_dir)

print(f"\n{'='*80}")
print("All visualizations complete!")
print(f"Output directory: {results_dir}")
print(f"{'='*80}\n")

print("Note:")
print("   - These are TRUE attention maps (Query-Key-Value Attention)")
print("   - Each pixel (i,j) shows how much query position i 'attends to' key position j")
print("   - Multiple attention heads learn different patterns")
print("   - Average attention map is the mean across all heads")
print("\nInterpreting attention maps:")
print("   - Bright diagonal: Each patch attends to itself")
print("   - Bright horizontal regions: Specific patches are attended by many patches")
print("   - Bright vertical regions: Specific patches attend to many patches")
