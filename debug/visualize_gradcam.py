"""
Grad-CAMを使ってMobile ViTの根拠領域を可視化

Grad-CAM (Gradient-weighted Class Activation Mapping) は、
ニューラルネットワークが予測時にどの領域に注目しているかを可視化する手法です。

使い方:
    python debug/visualize_gradcam.py

出力:
    - results/[timestamp]/gradcam_*.png: Grad-CAM可視化
    - results/[timestamp]/gradcam_overlay_*.png: 元画像とのオーバーレイ
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from models import HTRNet
from utils.htr_dataset import HTRDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datetime import datetime
import cv2

# 結果保存用ディレクトリ
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', f'gradcam_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Created results directory: {results_dir}\n")


class GradCAM:
    """
    Grad-CAM実装クラス
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: ニューラルネットワークモデル
            target_layer: Grad-CAMを適用するレイヤー（通常は最後の畳み込み層）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # フックを登録
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Forward/Backwardフックを登録"""

        def forward_hook(module, input, output):
            # Forward時の特徴マップを保存
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Backward時の勾配を保存
            self.gradients = grad_output[0].detach()

        # フックを登録
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)

        self.hook_handles.append(handle_forward)
        self.hook_handles.append(handle_backward)

    def remove_hooks(self):
        """登録したフックを削除"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_cam(self, input_image, use_sequence_average=True):
        """
        Grad-CAMを生成

        Args:
            input_image: 入力画像 (1, C, H, W)
            use_sequence_average: シーケンス全体の平均勾配を使用するか

        Returns:
            cam: Grad-CAMヒートマップ (H, W)
        """
        # Grad-CAMのためにtraining modeに設定（backwardを許可）
        # ただしDropoutやBatchNormは評価モードのまま
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                module.eval()

        # 勾配を有効化
        input_image.requires_grad_(True)

        # Forward pass
        output = self.model(input_image)

        # CTCの場合、outputは(seq_len, batch, nclasses)
        # または'both'の場合は(y_ctc, y_cnn)のタプル
        if isinstance(output, tuple):
            # 'both'モード: RNN出力を使用
            ctc_output = output[0]  # (seq_len, batch, nclasses)
        else:
            ctc_output = output

        if use_sequence_average:
            # シーケンス全体の予測スコアの合計をターゲットにする
            # 各時間ステップの最大確率クラスのスコアを合計
            probs = F.softmax(ctc_output, dim=2)  # (seq_len, batch, nclasses)
            max_probs, _ = probs.max(dim=2)  # (seq_len, batch)

            # 上位50%の時間ステップのスコアを使用（ブランクを避ける）
            sorted_probs, sorted_indices = max_probs[:, 0].sort(descending=True)
            top_k = max(1, len(sorted_probs) // 2)
            top_indices = sorted_indices[:top_k]

            # 上位スコアの合計
            target_score = max_probs[top_indices, 0].sum()
        else:
            # 単一の最高スコア時間ステップを使用
            max_probs = F.softmax(ctc_output, dim=2).max(dim=2)[0]  # (seq_len, batch)
            target_index = max_probs[:, 0].argmax().item()
            target_class = ctc_output[target_index, 0].argmax().item()
            target_score = ctc_output[target_index, 0, target_class]

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Grad-CAMの計算
        # gradients: (batch, channels, H, W)
        # activations: (batch, channels, H, W)
        gradients = self.gradients[0].to(self.activations.device)  # (channels, H, W)
        activations = self.activations[0]  # (channels, H, W)

        # Global Average Pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (channels,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU（負の値を0にする）
        cam = F.relu(cam)

        # 正規化 [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET, alpha=0.6):
    """
    画像にヒートマップをオーバーレイ

    Args:
        org_img: 元画像 (H, W) or (H, W, C)
        activation_map: アクティベーションマップ (H, W)
        colormap: カラーマップ
        alpha: 透明度（ヒートマップの不透明度、高いほど強調）

    Returns:
        overlay_img: オーバーレイ画像
        heatmap: ヒートマップ
    """
    # アクティベーションマップを元画像のサイズにバイリニア補間でリサイズ
    activation_map_resized = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

    # [0, 255]にスケール
    heatmap = np.uint8(255 * activation_map_resized)

    # カラーマップ適用
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 元画像をRGBに変換
    if len(org_img.shape) == 2:  # グレースケール
        org_img_rgb = np.stack([org_img] * 3, axis=-1)
    else:
        org_img_rgb = org_img

    # [0, 255]にスケール
    if org_img_rgb.max() <= 1.0:
        org_img_rgb = np.uint8(255 * org_img_rgb)

    # オーバーレイ（ヒートマップを強調）
    overlay = heatmap * alpha + org_img_rgb * (1 - alpha)
    overlay = np.uint8(overlay)

    return overlay, heatmap


def visualize_gradcam_for_sample(model, img, transcr, sample_idx, target_layers, results_dir, classes):
    """
    サンプルに対してGrad-CAMを可視化

    Args:
        model: モデル
        img: 入力画像 (1, C, H, W)
        transcr: 正解文字列
        sample_idx: サンプル番号
        target_layers: Grad-CAMを適用するレイヤーのリスト
        results_dir: 保存先ディレクトリ
        classes: 文字クラスのリスト
    """
    print(f"\nGenerating Grad-CAM for sample {sample_idx}...")

    # 元画像を取得
    img_np = img[0, 0].cpu().numpy()  # (H, W)

    # 予測を取得
    model.eval()
    with torch.no_grad():
        output = model(img)
        if isinstance(output, tuple):
            ctc_output = output[0]  # (seq_len, batch, nclasses)
        else:
            ctc_output = output

        # CTCデコード（簡易版）
        predicted_indices = torch.argmax(ctc_output[:, 0, :], dim=1)  # (seq_len,)
        decoded = []
        prev_idx = -1
        for idx in predicted_indices:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:  # 0はブランク
                if idx - 1 < len(classes):
                    decoded.append(classes[idx - 1])
            prev_idx = idx
        predicted_text = ''.join(decoded)

    print(f"  Ground Truth: '{transcr}'")
    print(f"  Predicted:    '{predicted_text}'")

    # 各ターゲットレイヤーに対してGrad-CAMを生成
    num_layers = len(target_layers)
    fig, axes = plt.subplots(3, num_layers, figsize=(7 * num_layers, 16))

    # 1行の場合の処理
    if num_layers == 1:
        axes = axes.reshape(-1, 1)

    for layer_idx, (layer_name, target_layer) in enumerate(target_layers.items()):
        print(f"  Processing {layer_name}...")

        # Grad-CAM生成
        gradcam = GradCAM(model, target_layer)

        try:
            cam = gradcam.generate_cam(img)

            # 統計情報を計算
            cam_mean = cam.mean()
            cam_max = cam.max()
            cam_min = cam.min()
            activation_ratio = (cam > 0.5).sum() / cam.size  # 50%以上の活性化領域の割合

            # 1行目: 元画像
            axes[0, layer_idx].imshow(img_np, cmap='gray')
            axes[0, layer_idx].set_title(
                f'{layer_name}\n'
                f'GT: "{transcr[:30]}..."\n'
                f'Pred: "{predicted_text[:30]}..."',
                fontsize=9
            )
            axes[0, layer_idx].axis('off')

            # 2行目: Grad-CAMヒートマップ
            im = axes[1, layer_idx].imshow(cam, cmap='jet', aspect='auto')
            axes[1, layer_idx].set_title(
                f'Grad-CAM Heatmap\n'
                f'Mean: {cam_mean:.3f}, Max: {cam_max:.3f}\n'
                f'Active: {activation_ratio*100:.1f}%',
                fontsize=9
            )
            axes[1, layer_idx].axis('off')
            plt.colorbar(im, ax=axes[1, layer_idx], fraction=0.046, pad=0.04)

            # 3行目: オーバーレイ
            overlay, heatmap = apply_colormap_on_image(img_np, cam, alpha=0.6)
            axes[2, layer_idx].imshow(overlay)
            axes[2, layer_idx].set_title(f'Overlay (alpha=0.6)', fontsize=9)
            axes[2, layer_idx].axis('off')

        except Exception as e:
            print(f"    Error generating Grad-CAM for {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            for row in range(3):
                axes[row, layer_idx].text(0.5, 0.5, f'Error:\n{str(e)[:50]}',
                                         ha='center', va='center',
                                         transform=axes[row, layer_idx].transAxes,
                                         fontsize=8)
                axes[row, layer_idx].axis('off')
        finally:
            gradcam.remove_hooks()

    plt.tight_layout()
    output_path = os.path.join(results_dir, f'gradcam_sample_{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


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

# Grad-CAMを適用するターゲットレイヤーを指定
# バックボーンの最後の畳み込み層を優先的に選択
print("Finding target layers for Grad-CAM...")
target_layers = {}

# 戦略: バックボーンの重要な位置の畳み込み層を選択
# 1. 最後のBasicBlock (cnv10) - バックボーン最終層
# 2. MobileViT2直前のBasicBlock (cnv6)
# 3. MobileViT1直前のBasicBlock (cnv2)

target_layer_candidates = [
    'features.features.cnv10',  # 最後のBasicBlock (256ch)
    'features.features.cnv6',   # MobileViT2直前 (128ch)
    'features.features.cnv2',   # MobileViT1直前 (64ch)
]

for name, module in net.named_modules():
    # BasicBlockを探す
    if any(candidate in name for candidate in target_layer_candidates):
        # BasicBlockのconv2（最終畳み込み）を使用
        if name.endswith('.conv2') and isinstance(module, torch.nn.Conv2d):
            # パス名を短縮
            short_name = name.replace('features.features.', '')
            target_layers[short_name] = module
            print(f"  Found target layer: {short_name} (shape: {module.out_channels}ch)")

if len(target_layers) == 0:
    print("ERROR: No suitable target layers found!")
    sys.exit(1)

print(f"  Total target layers: {len(target_layers)}")

# メイン処理
print("\n" + "="*80)
print("Visualizing Grad-CAM (Gradient-weighted Class Activation Mapping)")
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

    visualize_gradcam_for_sample(net, img, transcr, i+1, target_layers, results_dir, classes)

print(f"\n{'='*80}")
print("All visualizations complete!")
print(f"Output directory: {results_dir}")
print(f"{'='*80}\n")

print("Note:")
print("   - Grad-CAM shows which regions the model focuses on for predictions")
print("   - Brighter regions (red/yellow) indicate higher importance")
print("   - Darker regions (blue/purple) indicate lower importance")
print("   - Each target layer shows different feature levels:")
print("     - Earlier layers: Low-level features (edges, textures)")
print("     - Later layers: High-level features (characters, words)")
