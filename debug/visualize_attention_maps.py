"""
Mobile ViTのアテンションマップを可視化するスクリプト

使い方:
    python debug/visualize_attention_maps.py

出力:
    - results/[timestamp]/attention_map_*.png: アテンションマップの可視化
    - results/[timestamp]/attention_stats_*.txt: アテンション統計情報
"""
import sys
import os

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from models import HTRNet
from utils.htr_dataset import HTRDataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 結果保存用ディレクトリ作成
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', f'attention_maps_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Created results directory: {results_dir}\n")

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

# 学習済み重みのロード（オプション）
if config.resume and os.path.exists(config.resume):
    print(f"Loading checkpoint: {config.resume}")
    load_dict = torch.load(config.resume, map_location='cpu')
    net.load_state_dict(load_dict, strict=False)
    print("Loaded checkpoint\n")

net.to(device)
net.eval()


# アテンションマップを取得するためのフック
class AttentionMapExtractor:
    """MultiHeadAttentionのアテンション重みを取得するクラス"""

    def __init__(self):
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self, model):
        """モデルのMobileViTBlock内のTransformerにフックを登録"""

        def get_attention_hook(name):
            def hook(module, input, output):
                # MultiHeadAttentionの出力: (output, attention_weights)
                # ただし、need_weights=Trueが必要
                # TransformerEncoderLayerは内部的にneed_weights=Falseを使用するため、
                # 直接self_attnにアクセスする
                pass
            return hook

        # MobileViTBlockを探す
        for name, module in model.named_modules():
            if 'mvit' in name and hasattr(module, 'transformer'):
                # TransformerEncoderにアクセス
                print(f"Found MobileViT block: {name}")
                for layer_idx, layer in enumerate(module.transformer.layers):
                    # self_attnにフックを登録
                    hook = self._create_hook(f"{name}_layer{layer_idx}")
                    handle = layer.self_attn.register_forward_hook(hook)
                    self.hooks.append(handle)
                    print(f"  Registered hook on layer {layer_idx}")

    def _create_hook(self, name):
        """特定のレイヤー用のフックを作成"""
        def hook(module, input, output):
            # MultiHeadAttentionの出力: (attn_output, attn_output_weights)
            # ただし、need_weightsがFalseの場合、attn_output_weightsはNone
            # 強制的にアテンション重みを計算させる必要がある

            # inputから直接計算（これは簡易版）
            # 実際のアテンション重みは取得が難しいので、出力のみ保存
            if len(output) > 1 and output[1] is not None:
                # アテンション重みが利用可能な場合
                self.attention_maps[name] = output[1].detach().cpu()
            else:
                # アテンション重みが利用できない場合は出力を保存
                self.attention_maps[name] = output[0].detach().cpu()
        return hook

    def remove_hooks(self):
        """登録したフックを削除"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


# より簡単なアプローチ：MobileViTBlockのforwardを修正してアテンション重みを返す
# これには、モデルのコードを一時的に修正する必要があります

def visualize_attention_simple(model, img, transcr, save_prefix, device):
    """
    より簡単なアプローチ：Transformer内部の中間出力を可視化

    アテンション重みの代わりに、各パッチの特徴量の変化を可視化します
    """

    with torch.no_grad():
        # 特徴量抽出
        y = model.features(img)

        # MobileViTブロックの前後で特徴量を取得
        # model.features.featuresを順番に処理
        intermediate_features = []
        x_temp = img

        for i, layer in enumerate(model.features.features):
            x_temp = layer(x_temp)

            # MobileViTブロックの場合、入力と出力を保存
            if 'mvit' in str(type(layer)).lower():
                intermediate_features.append({
                    'layer_name': f'MobileViT_{i}',
                    'input_shape': x_temp.shape,
                    'output': x_temp.detach().cpu().clone()
                })

        return intermediate_features


def visualize_patch_attention(model, img, transcr, save_prefix, device):
    """
    パッチごとのアテンションを疑似的に可視化

    MobileViTは画像をパッチに分割してTransformerに入力します。
    各パッチの重要度を、最終的な特徴量の大きさで可視化します。
    """

    with torch.no_grad():
        # 1. 特徴量抽出
        y = model.features(img)
        B, C, H, W = y.shape

        # 2. 各MobileViTブロックの出力を取得
        x_temp = img
        mvit_outputs = []

        for i, layer in enumerate(model.features.features):
            x_temp_before = x_temp.clone()
            x_temp = layer(x_temp)

            # MobileViTブロックの場合
            if hasattr(layer, 'transformer'):
                # パッチサイズを取得
                p = layer.p
                B_temp, C_temp, H_temp, W_temp = x_temp.shape

                # 特徴量の変化を計算（各位置での差分のL2ノルム）
                # これがアテンションの疑似的な指標になります
                diff = (x_temp - x_temp_before).pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)

                # L2ノルムで正規化して重要度マップを作成
                importance_map = diff / (diff.max() + 1e-8)

                mvit_outputs.append({
                    'layer_idx': i,
                    'layer_name': f'MobileViT_layer{i}',
                    'patch_size': p,
                    'importance_map': importance_map.cpu().numpy(),
                    'shape': x_temp.shape
                })

        # 3. 可視化
        num_mvit_blocks = len(mvit_outputs)
        if num_mvit_blocks == 0:
            print("WARNING: No MobileViT blocks found!")
            return

        fig, axes = plt.subplots(1, num_mvit_blocks + 1, figsize=(6 * (num_mvit_blocks + 1), 5))
        if num_mvit_blocks == 0:
            axes = [axes]

        # 元画像を表示
        img_np = img[0, 0].cpu().numpy()
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f'Input Image\n"{transcr}"', fontsize=12)
        axes[0].axis('off')

        # 各MobileViTブロックの重要度マップを表示
        for idx, mvit_data in enumerate(mvit_outputs):
            importance = mvit_data['importance_map'][0, 0]  # (H, W)

            im = axes[idx + 1].imshow(importance, cmap='hot', aspect='auto')
            axes[idx + 1].set_title(
                f"{mvit_data['layer_name']}\n"
                f"Patch size: {mvit_data['patch_size']}x{mvit_data['patch_size']}\n"
                f"Shape: {mvit_data['shape'][2]}x{mvit_data['shape'][3]}",
                fontsize=10
            )
            axes[idx + 1].axis('off')
            plt.colorbar(im, ax=axes[idx + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        output_path = os.path.join(results_dir, f'{save_prefix}_patch_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved patch importance map: {output_path}")

        # 統計情報を保存
        stats_path = os.path.join(results_dir, f'{save_prefix}_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Ground Truth: {transcr}\n")
            f.write(f"Number of MobileViT blocks: {num_mvit_blocks}\n\n")

            for mvit_data in mvit_outputs:
                importance = mvit_data['importance_map'][0, 0]
                f.write(f"{mvit_data['layer_name']}:\n")
                f.write(f"  Patch size: {mvit_data['patch_size']}x{mvit_data['patch_size']}\n")
                f.write(f"  Feature map shape: {mvit_data['shape']}\n")
                f.write(f"  Importance - Mean: {importance.mean():.4f}, Max: {importance.max():.4f}, Min: {importance.min():.4f}\n")
                f.write("\n")

        print(f"Saved statistics: {stats_path}")


# メイン処理
print("="*80)
print("Visualizing MobileViT Attention Maps")
print("="*80)

# サンプル選択
num_samples = 1
sample_indices = [0]

for i, idx in enumerate(sample_indices):
    img, transcr = dataset[idx]
    img = img.unsqueeze(0).to(device)

    print(f"\n{'='*80}")
    print(f"Sample {i+1}/{num_samples} (Index: {idx})")
    print(f"{'='*80}")
    print(f"Ground Truth: '{transcr}'")

    # パッチ重要度マップを可視化
    visualize_patch_attention(net, img, transcr, f'sample_{i+1}', device)

print(f"\n{'='*80}")
print("All visualizations complete!")
print(f"Output directory: {results_dir}")
print(f"{'='*80}\n")

print("Note:")
print("   - Visualized 'patch importance maps' for each MobileViT block")
print("   - Importance is calculated from L2 norm of input-output difference")
print("   - Red areas: Regions heavily modified by Transformer (important regions)")
print("   - Blue areas: Regions with little change")
print("   - This visualization approximates how much each patch is 'attended to'")
print("\nTo get true attention weights:")
print("   - Use visualize_true_attention.py script instead")
print("   - That script directly captures Query-Key-Value attention weights")
