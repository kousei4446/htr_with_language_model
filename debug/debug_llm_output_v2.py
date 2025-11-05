"""
学習済みモデルのCTC出力とLLM出力を比較するスクリプト (新実装対応版)
- 新しいMobileViT+LLM実装に対応
- 旧チェックポイントも読み込み可能
"""
import sys
import os

# 親ディレクトリをパスに追加（modelsとutilsをimportするため）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from models import HTRNet
from utils.htr_dataset import HTRDataset
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# results/debug_llm/日付-時刻/ディレクトリを作成（親ディレクトリに）
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'debug_llm', timestamp)
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
    'test',  # テストデータで確認
    fixed_size=(config.preproc.image_height, config.preproc.image_width)
)

# 学習時の文字クラスを読み込み（data_path/classes.npyから）
classes_path = os.path.join(data_path, 'classes.npy')
classes = np.load(classes_path, allow_pickle=True).tolist()
print(f"Character classes: {len(classes)} different characters (loaded from training)")


def decode_ctc(tokens, char_classes):
    """
    CTCデコード: 重複削除とblank除去

    Args:
        tokens: (seq_len,) のトークンID配列
        char_classes: 文字リスト

    Returns:
        デコードされた文字列
    """
    result = []
    prev = -1
    for t in tokens:
        if t != prev and t != 0:  # 0はblank
            if t - 1 < len(char_classes):
                result.append(char_classes[t - 1])
        prev = t
    return ''.join(result)


# モデル作成（LLM有効）
print("\n🔧 Creating model with LLM enabled...")
llm_source = config.train.get('llm_source', 'rnn')
print(f"   LLM source: {llm_source}")

net = HTRNet(config.arch, len(classes) + 1, use_llm=True, llm_source=llm_source)

# 学習済み重みをロード（親ディレクトリから）
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'saved_models', 'CRNN+LLM', '700.pt')
print(f"\n📥 Loading checkpoint: {model_path}")

if not os.path.exists(model_path):
    print(f"❌ Error: Checkpoint not found at {model_path}")
    print("   Please update the model_path in this script")
    sys.exit(1)

load_dict = torch.load(model_path, map_location='cpu')

# 旧実装のLLM関連キーをフィルタリング（新実装との互換性のため）
filtered_dict = {}
llm_keys_filtered = 0

for k, v in load_dict.items():
    # 旧実装のキー（top.connector.*, top.llm.*）を除外
    if k.startswith('top.connector.') or k.startswith('top.llm.'):
        llm_keys_filtered += 1
        continue
    filtered_dict[k] = v

print(f"   Filtered {llm_keys_filtered} LLM-related keys from checkpoint")

# モデルにロード（strict=False: 新しいconnector_rnn等は初期化される）
missing_keys, unexpected_keys = net.load_state_dict(filtered_dict, strict=True)

print(f"✅ Loaded checkpoint successfully")
print(f"   Loaded: {len(filtered_dict)} parameters")
if missing_keys:
    print(f"   Missing keys: {len(missing_keys)} (new LLM components, initialized randomly)")
if unexpected_keys:
    print(f"   Unexpected keys: {len(unexpected_keys)}")

net.to(device)
net.eval()

# Connector選択
print("\n🔧 Selecting Connector based on llm_source...")
connector = None
connector_name = None

if hasattr(net.top, 'connector_rnn') and net.top.connector_rnn is not None:
    connector = net.top.connector_rnn
    connector_name = 'RNN'
    print(f"   Using Connector_RNN (512 → 3072)")
elif hasattr(net.top, 'connector_mv1') and net.top.connector_mv1 is not None:
    connector = net.top.connector_mv1
    connector_name = 'MobileViT1'
    print(f"   Using Connector_MV1 (64 → 3072)")
elif hasattr(net.top, 'connector_mv2') and net.top.connector_mv2 is not None:
    connector = net.top.connector_mv2
    connector_name = 'MobileViT2'
    print(f"   Using Connector_MV2 (128 → 3072)")
else:
    print("❌ Error: No connector available!")
    print("   Check use_llm and llm_source in config.yaml")
    sys.exit(1)

# LLMチェック
if not hasattr(net.top, 'llm') or net.top.llm is None:
    print("❌ Error: LLM not available!")
    print("   Set use_llm: true in config.yaml")
    sys.exit(1)

print(f"✅ LLM available (shared across all paths)")

# サンプル画像で確認
print("\n🔍 Testing CTC vs LLM output on sample images...\n")
print("="*80)

num_samples = 5
indices = [5, 37, 12, 4, 67]  # 固定サンプルインデックス

for i, idx in enumerate(indices):
    img, transcr = dataset[idx]
    img_display = img.clone()  # 表示用に保存
    img = img.unsqueeze(0).to(device)  # (1, 1, 128, 1024)

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
        y = net.features(img_feat)  # (1, 256, 1, width)

        # 2. RNN処理
        y_seq = y.permute(2, 3, 0, 1)[0]  # (width, 1, 256)
        y1 = net.top.rec1(y_seq)[0]  # (width, 1, 512)

        if net.top.recN is not None:
            y_rnn = net.top.recN(y1)[0]  # (width, 1, 512)
        else:
            y_rnn = y1

        # === CTC出力 ===
        y_ctc = net.top.fnl(y_rnn)  # (width, 1, nclasses)
        ctc_tokens = torch.argmax(y_ctc, dim=-1).squeeze().cpu().numpy()  # (width,)
        ctc_text = decode_ctc(ctc_tokens, classes)
        print(f"🔤 CTC Prediction: '{ctc_text}'")

        # === LLM出力 ===
        # Connector入力準備（RNN layer1出力を使用）
        prefix_input = y1.permute(1, 0, 2)  # (1, width, 512)
        inputs_embeds = connector(prefix_input)  # (1, num_tokens, 3072)

        seq_len = inputs_embeds.shape[1]
        print(f"🔧 Connector output tokens: {seq_len} (using {connector_name} path)")

        # Ground Truth文字列をトークン化
        llm_labels = net.top.llm.tokenizer(
            [transcr],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len
        )
        labels = llm_labels["input_ids"].to(device)  # (1, seq_len)

        # LLM順伝播
        output_llm = net.top.llm.model(
            inputs_embeds=inputs_embeds.half(),
            labels=labels
        )

        # LLM予測をデコード
        logits = output_llm.logits  # (1, seq_len, vocab_size)
        preds = torch.argmax(logits, dim=-1)  # (1, seq_len)
        pred_tokens = preds[0].cpu().numpy().tolist()
        llm_text = net.top.llm.tokenizer.decode(pred_tokens, skip_special_tokens=True)
        print(f"🤖 LLM Prediction: '{llm_text}'")

    # 個別画像として保存
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.imshow(img_display.squeeze(), cmap='gray')
    ax.set_title(
        f"Sample {i+1} (Index: {idx}) - Using {connector_name} Connector\n"
        f"GT:  '{transcr}'\n"
        f"CTC: '{ctc_text}'\n"
        f"LLM: '{llm_text}'",
        fontsize=11,
        loc='left'
    )
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(results_dir, f'sample_{i+1}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved: {output_path}")

print(f"\n{'='*80}")
print("✅ Done!")
print(f"📊 Images saved to: {results_dir}")
print(f"{'='*80}")

print("\n💡 Note:")
print("   - 新しいMobileViT+LLM実装に対応")
print(f"   - Connector: {connector_name} path")
print("   - CTC: 従来のCTCデコーダの出力")
print("   - LLM: Connectorを通してLLMで生成した出力")
print("   - 両者の精度を比較できます")
