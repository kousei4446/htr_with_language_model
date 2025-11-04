"""
Connector層の出力と正解文字列のLLM embeddingsの潜在空間分布を可視化
"""
import sys
import os

# 親ディレクトリをパスに追加（modelsとutilsをimportするため）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from models import HTRNet
from utils.htr_dataset import HTRDataset
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as F
from datetime import datetime

# results/日付-時刻/ディレクトリを作成（親ディレクトリに）
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', timestamp)
os.makedirs(results_dir, exist_ok=True)
print(f"📁 Created/verified results directory: {results_dir}")

# 設定ロード（親ディレクトリから）
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
config = OmegaConf.load(config_path)

device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# データセット準備
print("\n📂 Loading dataset...")

# config.data.pathの相対パスをプロジェクトルートから解決
data_path = config.data.path
if not os.path.isabs(data_path):
    # config.yamlがある親ディレクトリ（プロジェクトルート）からの相対パス
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, data_path)

dataset = HTRDataset(
    data_path,
    'test',
    fixed_size=(config.preproc.image_height, config.preproc.image_width)
)

# 学習時の文字クラスを読み込み（saved_models/classes.npyから）
classes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'saved_models', 'classes.npy')
classes = np.load(classes_path, allow_pickle=True).tolist()
print(f"Character classes: {len(classes)} different characters (loaded from training)")

# モデル作成（LLM有効）
print("\n🔧 Creating model with LLM enabled...")
net = HTRNet(config.arch, len(classes) + 1, use_llm=True)

# 学習済み重みをロード（親ディレクトリから）
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'saved_models', '10-30_llmmobilevit', '100.pt')
print(f"\n📥 Loading checkpoint: {model_path}")
load_dict = torch.load(model_path, map_location='cpu')
missing_keys, unexpected_keys = net.load_state_dict(load_dict, strict=True)

print(f"✅ Loaded checkpoint successfully")

if missing_keys:
    print(f"   Missing keys: {len(missing_keys)}")
if unexpected_keys:
    print(f"   Unexpected keys: {len(unexpected_keys)}")

net.to(device)
net.eval()

# サンプル選択
print("\n🔍 Analyzing latent space distribution...\n")
print("="*80)

num_samples = 5
indices = [5,37,12,4,67]  # 固定サンプルインデックス

for i, idx in enumerate(indices):
    img, transcr = dataset[idx]
    img = img.unsqueeze(0).to(device)

    print(f"\n{'='*80}")
    print(f"Sample {i+1}/{num_samples} (Index: {idx})")
    print(f"{'='*80}")
    print(f"📝 Ground Truth: '{transcr}'")

    with torch.no_grad():
        # 1. 特徴量抽出
        if net.stn is not None:
            img_feat = net.stn(img)
        else:
            img_feat = img
        y = net.features(img_feat)

        # 2. RNN layer1出力取得
        y_seq = y.permute(2, 3, 0, 1)[0]
        y1 = net.top.rec1(y_seq)[0]

        # 3. Connector出力取得 (128, 3072)
        prefix_input = y1.permute(1, 0, 2)
        connector_output = net.top.connector(prefix_input).squeeze(0)  # (128, 3072)

        # 4. 正解文字列のLLM embeddings取得 (seq_len, 3072)
        tokens = net.top.llm.tokenizer(
            [transcr],  # リストでラップ
            return_tensors="pt",
            padding=False,  # パディングなし（実際のトークン数を取得）
            truncation=False  # 切り詰めなし
        )
        token_ids = tokens.input_ids.to(device)
        gt_embeddings = net.top.llm.model.model.embed_tokens(token_ids).squeeze(0)  # (seq_len, 3072)

        print(f"🔧 Connector output shape: {connector_output.shape}")
        print(f"🔧 GT embeddings shape: {gt_embeddings.shape}")

        # === 可視化1: t-SNE散布図 ===
        print("📊 Generating t-SNE visualization...")
        all_vectors = torch.cat([connector_output, gt_embeddings], dim=0).cpu().numpy()

        # t-SNEで2次元に削減
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vectors)-1))
        reduced = tsne.fit_transform(all_vectors)

        num_connector_tokens = connector_output.shape[0]
        connector_2d = reduced[:num_connector_tokens]
        gt_2d = reduced[num_connector_tokens:]

        # プロット
        plt.figure(figsize=(12, 8))
        plt.scatter(connector_2d[:, 0], connector_2d[:, 1],
                   c='blue', marker='o', s=50, alpha=0.6, label=f'Connector output ({num_connector_tokens} tokens)')
        plt.scatter(gt_2d[:, 0], gt_2d[:, 1],
                   c='red', marker='x', s=100, alpha=0.8, label=f'GT embeddings ({len(gt_2d)} tokens)')
        plt.legend(fontsize=12)
        plt.title(f"Latent Space Distribution (t-SNE)\nGT: '{transcr}'", fontsize=14)
        plt.xlabel('t-SNE dimension 1', fontsize=12)
        plt.ylabel('t-SNE dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'latent_space_sample_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # === 可視化2: コサイン類似度ヒートマップ ===
        print("📊 Generating cosine similarity heatmap...")
        similarity = F.cosine_similarity(
            connector_output.unsqueeze(1),  # (128, 1, 3072)
            gt_embeddings.unsqueeze(0),     # (1, seq_len, 3072)
            dim=2  # (128, seq_len)
        ).cpu().numpy()

        plt.figure(figsize=(max(10, len(gt_embeddings)), 10))
        im = plt.imshow(similarity, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Cosine Similarity')
        plt.xlabel(f'GT tokens ({len(gt_embeddings)})', fontsize=12)
        plt.ylabel('Connector tokens (128)', fontsize=12)
        plt.title(f"Cosine Similarity Heatmap\nGT: '{transcr}'", fontsize=14)

        # 統計情報を追加
        mean_sim = similarity.mean()
        max_sim = similarity.max()
        min_sim = similarity.min()
        plt.text(0.02, 0.98, f'Mean: {mean_sim:.3f}\nMax: {max_sim:.3f}\nMin: {min_sim:.3f}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'similarity_heatmap_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # === 可視化3: ノルム分布ヒストグラム ===
        print("📊 Generating norm distribution histogram...")
        connector_norms = torch.norm(connector_output, dim=1).cpu().numpy()
        gt_norms = torch.norm(gt_embeddings, dim=1).cpu().numpy()

        plt.figure(figsize=(12, 6))
        plt.hist(connector_norms, bins=30, alpha=0.6, label='Connector output', color='blue', edgecolor='black')
        plt.hist(gt_norms, bins=30, alpha=0.6, label='GT embeddings', color='red', edgecolor='black')
        plt.axvline(connector_norms.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Connector mean: {connector_norms.mean():.2f}')
        plt.axvline(gt_norms.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'GT mean: {gt_norms.mean():.2f}')
        plt.xlabel('L2 Norm', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=11)
        plt.title(f'Distribution of Vector Norms\nGT: \'{transcr}\'', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'norm_distribution_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved visualizations to {results_dir}")
        print(f"   - latent_space_sample_{i+1}.png")
        print(f"   - similarity_heatmap_{i+1}.png")
        print(f"   - norm_distribution_{i+1}.png")

print(f"\n{'='*80}")
print("✅ All visualizations complete!")
print(f"📊 Total files created: {num_samples * 3} images")
print(f"📂 Output directory: {results_dir}")
print(f"{'='*80}")

print("\n💡 Note:")
print("   - 学習済みモデル（700 epoch, LLM込み）を使用")
print("   - 青色: Connector出力 (可変長トークン, 3072次元)")
print("   - 赤色: 正解embeddings (可変長, 3072次元)")
print("   - t-SNEで潜在空間の分布を可視化")
print("   - コサイン類似度で両者の対応関係を分析")
print("   - 学習が進んでいる場合、青と赤が近づいているはず")
