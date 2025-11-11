from omegaconf import OmegaConf

import sys
import os
import numpy as np
import torch
from utils.htr_dataset import HTRDataset
from utils.metrics import CER, WER

from models import HTRNet


def parse_args():
    # デフォルトの設定ファイルパスを設定
    if len(sys.argv) < 2:
        config_path = "config.yaml"  # デフォルトのパス
        print(f"No config file specified, using default: {config_path}")
    else:
        config_path = sys.argv[1]

    conf = OmegaConf.load(config_path)

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


def decode(net, img, classes):
    """CTC出力をデコードして文字列に変換"""
    with torch.no_grad():
        output = net(img)
        output = output[0]
    predicted_indices = torch.argmax(output, dim=2).squeeze()
    decoded = []
    prev_idx = -1
    for idx in predicted_indices:
        idx = idx.item()
        if idx != 0 and idx != prev_idx:  # 0はブランク、連続文字はスキップ
            decoded.append(classes[idx-1])
        prev_idx = idx

    predicted_text = ''.join(decoded)
    return predicted_text


# =============================================================================
# メイン処理
# =============================================================================
config = parse_args()
dataset_folder = config.data.path

classes = np.load(os.path.join(dataset_folder, 'classes.npy'))

cdict = {c:(i+1) for i,c in enumerate(classes)}
icdict = {(i+1):c for i,c in enumerate(classes)}

classes = {
'classes': classes,
'c2i': cdict,
'i2c': icdict
}

classes = classes['classes']

net = HTRNet(config.arch, len(classes) + 1)

if config.resume is not None:
    load_dict = torch.load(config.resume)
    load_status = net.load_state_dict(load_dict, strict=False)


device = config.device
net.to(device)

net.eval()
fixed_size = (config.preproc.image_height, config.preproc.image_width)

# HTRDatasetを正しく初期化
dataset = HTRDataset(config.data.path, "test", fixed_size=fixed_size, transforms=None)

IMG_IDX = 1
img, label = dataset[IMG_IDX]
img = img.unsqueeze(0).to(device)


# =============================================================================
# CTC予測の実行
# =============================================================================
print("\n" + "="*70)
print("CTC Prediction")
print("="*70 + "\n")

ctc_predict = decode(net, img, classes)

print("Prediction Results:")
print("-" * 70)
print(f"Ground Truth   : {label}")
print(f"CTC Prediction : {ctc_predict}")
print("-" * 70)

# CERとWERの計算
cer = CER()
wer = WER(mode='tokenizer')

cer.update(ctc_predict, label)
wer.update(ctc_predict, label)

cer_score = cer.score()
wer_score = wer.score()

print(f"\nMetrics:")
print(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
print(f"  WER: {wer_score:.4f} ({wer_score*100:.2f}%)")
print("="*70 + "\n")




words = [label,ctc_predict]


# 必要ライブラリ（まだ入れてなければ）
# pip install transformers torch


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) モデルとトークナイザを準備
model_name = "gpt2"  # gpt2-small
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for word in words:
    # 2) トークン化（注意：GPT-2は先頭に自動でスペシャルトークンを入れない）
    enc = tokenizer(word, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # shape (1, seq_len)

    # 3) モデルに入力して logits を得る
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape (1, seq_len, vocab_size)

    # 4) CEロスの計算（シーケンス全体）
    # 入力シーケンスの各位置で次のトークンを予測
    # logits[:, :-1, :] で位置0～seq_len-2の予測
    # input_ids[:, 1:] で位置1～seq_len-1の正解ラベル
    shift_logits = logits[:, :-1, :].contiguous()  # (1, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:].contiguous()    # (1, seq_len-1)

    # CEロスを計算
    ce_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # (seq_len-1, vocab_size)
        shift_labels.view(-1)                           # (seq_len-1,)
    )

    print("\n" + "="*70)
    print("Language Model Loss")
    print("="*70)
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    print(f"Perplexity: {torch.exp(ce_loss).item():.4f}")
    print("="*70 + "\n")

    # 5) 各位置でのCEロスを表示
    print("Per-position CE Loss:")
    print("-" * 70)
    for i in range(shift_logits.size(1)):
        pos_logits = shift_logits[0, i, :]  # (vocab_size,)
        pos_label = shift_labels[0, i].item()
        pos_loss = torch.nn.functional.cross_entropy(pos_logits, shift_labels[0, i])
        pred_token = tokenizer.decode([input_ids[0, i].item()])
        true_token = tokenizer.decode([pos_label])
        print(f"  Position {i}: loss={pos_loss.item():.4f}  input={repr(pred_token):8}  target={repr(true_token):8}")
    print("="*70 + "\n")










# Ground Truth   : I don't think he will storm the charts with this one, but it's a good start.
# CTC Prediction : I don't thin he will storm the charks with this one, but it's a good start.
# Language Model Loss
# ======================================================================
# Cross Entropy Loss: 4.6952
# Perplexity: 109.4232
# ======================================================================

# Per-position CE Loss:
# ----------------------------------------------------------------------
#   Position 0: loss=5.5552  input=' I'      target=' don'
#   Position 1: loss=0.0018  input=' don'    target="'t"
#   Position 2: loss=13.8591  input="'t"      target=' thin'
#   Position 3: loss=8.7752  input=' thin'   target=' he'
#   Position 4: loss=4.5540  input=' he'     target=' will'
#   Position 5: loss=11.5698  input=' will'   target=' storm'
#   Position 6: loss=2.7734  input=' storm'  target=' the'
#   Position 7: loss=10.0174  input=' the'    target=' char'
#   Position 8: loss=5.8111  input=' char'   target='ks'
#   Position 9: loss=3.5511  input='ks'      target=' with'
#   Position 10: loss=5.1356  input=' with'   target=' this'
#   Position 11: loss=3.4296  input=' this'   target=' one'
#   Position 12: loss=2.2281  input=' one'    target=','
#   Position 13: loss=1.7502  input=','       target=' but'
#   Position 14: loss=2.6139  input=' but'    target=' it'
#   Position 15: loss=1.3981  input=' it'     target="'s"
#   Position 16: loss=2.0917  input="'s"      target=' a'
#   Position 17: loss=2.2039  input=' a'      target=' good'
#   Position 18: loss=4.7731  input=' good'   target=' start'
#   Position 19: loss=0.9640  input=' start'  target='.'
#   Position 20: loss=5.5433  input='.'       target=' '
# ======================================================================





# これlabel

# Cross Entropy Loss: 3.3369
# Perplexity: 28.1311
# ======================================================================

# Per-position CE Loss:
# ----------------------------------------------------------------------
#   Position 0: loss=5.5552  input=' I'      target=' don'
#   Position 1: loss=0.0018  input=' don'    target="'t"
#   Position 2: loss=1.4531  input="'t"      target=' think'
#   Position 3: loss=2.8709  input=' think'  target=' he'
#   Position 4: loss=3.6396  input=' he'     target=' will'
#   Position 5: loss=11.5414  input=' will'   target=' storm'
#   Position 6: loss=1.4286  input=' storm'  target=' the'
#   Position 7: loss=10.6680  input=' the'    target=' charts'
#   Position 8: loss=3.0879  input=' charts'  target=' with'
#   Position 9: loss=3.0238  input=' with'   target=' this'
#   Position 10: loss=2.3847  input=' this'   target=' one'
#   Position 11: loss=1.8599  input=' one'    target=','
#   Position 12: loss=0.8377  input=','       target=' but'
#   Position 13: loss=2.3756  input=' but'    target=' it'
#   Position 14: loss=1.0492  input=' it'     target="'s"
#   Position 15: loss=1.9181  input="'s"      target=' a'
#   Position 16: loss=2.0957  input=' a'      target=' good'
#   Position 17: loss=3.2440  input=' good'   target=' start'
#   Position 18: loss=0.7603  input=' start'  target='.'
#   Position 19: loss=6.9419  input='.'       target=' '
# ======================================================================












# ctc

# Cross Entropy Loss: 6.2707
# Perplexity: 528.8704
# ======================================================================

# Per-position CE Loss:
# ----------------------------------------------------------------------
#   Position 0: loss=3.0516  input=' Become'  target=' a'
#   Position 1: loss=8.1602  input=' a'      target=' success'
#   Position 2: loss=3.3830  input=' success'  target=' with'
#   Position 3: loss=2.5270  input=' with'   target=' a'
#   Position 4: loss=14.1273  input=' a'      target=' dise'
#   Position 5: loss=13.4479  input=' dise'   target=' and'
#   Position 6: loss=12.4961  input=' and'    target=' hey'
#   Position 7: loss=3.4998  input=' hey'    target=' prest'
#   Position 8: loss=0.1196  input=' prest'  target='o'
#   Position 9: loss=2.8872  input='o'       target='.'
#   Position 10: loss=2.9797  input='.'       target=' You'
#   Position 11: loss=2.0198  input=' You'    target="'re"
#   Position 12: loss=2.7722  input="'re"     target=' a'
#   Position 13: loss=6.7936  input=' a'      target=' st'
#   Position 14: loss=7.4775  input=' st'     target='arm'
#   Position 15: loss=5.5147  input='arm'     target='.'
#   Position 16: loss=7.5361  input='.'       target=' R'
#   Position 17: loss=7.7785  input=' R'      target='olly'
#   Position 18: loss=11.6799  input='olly'    target=' sings'
#   Position 19: loss=3.9030  input=' sings'  target=' with'
#   Position 20: loss=9.5309  input=' with'   target=' '
# ======================================================================


# ======================================================================
# Language Model Loss
# ======================================================================
# Cross Entropy Loss: 5.2461
# Perplexity: 189.8240
# ======================================================================

# Per-position CE Loss:
# ----------------------------------------------------------------------
#   Position 0: loss=3.0516  input=' Become'  target=' a'
#   Position 1: loss=8.1602  input=' a'      target=' success'
#   Position 2: loss=3.3830  input=' success'  target=' with'
#   Position 3: loss=2.5270  input=' with'   target=' a'
#   Position 4: loss=10.0859  input=' a'      target=' disc'
#   Position 5: loss=3.8503  input=' disc'   target=' and'
#   Position 6: loss=11.7640  input=' and'    target=' hey'
#   Position 7: loss=1.3798  input=' hey'    target=' prest'
#   Position 8: loss=0.0093  input=' prest'  target='o'
#   Position 9: loss=1.8329  input='o'       target='!'
#   Position 10: loss=1.7296  input='!'       target=' You'
#   Position 11: loss=1.6729  input=' You'    target="'re"
#   Position 12: loss=2.8932  input="'re"     target=' a'
#   Position 13: loss=4.5672  input=' a'      target=' star'
#   Position 14: loss=7.4268  input=' star'   target='....'
#   Position 15: loss=8.4851  input='....'    target=' R'
#   Position 16: loss=7.7746  input=' R'      target='olly'
#   Position 17: loss=11.0224  input='olly'    target=' sings'
#   Position 18: loss=3.5667  input=' sings'  target=' with'
#   Position 19: loss=9.7394  input=' with'   target=' '
# ======================================================================


# (htr_best) C:\Users\user\ai\HTR-best-practices>





# IDX＝１


# ======================================================================
# Language Model Loss
# ======================================================================
# Cross Entropy Loss: 5.2664
# Perplexity: 193.7087
# ======================================================================

# Per-position CE Loss:
# ----------------------------------------------------------------------
#   Position 0: loss=9.2400  input=' assured'  target='ness'
#   Position 1: loss=7.5195  input='ness'    target=' "'
#   Position 2: loss=8.0281  input=' "'      target='B'
#   Position 3: loss=6.5066  input='B'       target='ella'
#   Position 4: loss=5.6400  input='ella'    target=' Bella'
#   Position 5: loss=9.5795  input=' Bella'  target=' Marie'
#   Position 6: loss=1.0515  input=' Marie'  target='"'
#   Position 7: loss=2.5739  input='"'       target=' ('
#   Position 8: loss=5.4559  input=' ('      target='P'
#   Position 9: loss=6.1984  input='P'       target='arl'
#   Position 10: loss=1.6446  input='arl'     target='ophone'
#   Position 11: loss=2.6581  input='ophone'  target='),'
#   Position 12: loss=2.9837  input='),'      target=' a'
#   Position 13: loss=8.3723  input=' a'      target=' lively'
#   Position 14: loss=4.4655  input=' lively'  target=' song'
#   Position 15: loss=2.5332  input=' song'   target=' that'
#   Position 16: loss=6.4391  input=' that'   target=' changes'
#   Position 17: loss=4.9119  input=' changes'  target=' tempo'
#   Position 18: loss=7.0356  input=' tempo'  target=' mid'
#   Position 19: loss=0.2066  input=' mid'    target='-'
#   Position 20: loss=4.7615  input='-'       target='way'
#   Position 21: loss=5.5254  input='way'     target='.'
#   Position 22: loss=7.7952  input='.'       target=' '
# ======================================================================


# ======================================================================
# Language Model Loss
# ======================================================================
# Cross Entropy Loss: 6.9152
# Perplexity: 1007.5184
# ======================================================================

# Per-position CE Loss:
# ----------------------------------------------------------------------
#   Position 0: loss=9.2400  input=' assured'  target='ness'
#   Position 1: loss=7.5195  input='ness'    target=' "'
#   Position 2: loss=8.0281  input=' "'      target='B'
#   Position 3: loss=12.3432  input='B'       target='elta'
#   Position 4: loss=12.9170  input='elta'    target=' Bella'
#   Position 5: loss=10.0712  input=' Bella'  target=' Marie'
#   Position 6: loss=0.7873  input=' Marie'  target='"'
#   Position 7: loss=2.5367  input='"'       target=' ('
#   Position 8: loss=5.5375  input=' ('      target='P'
#   Position 9: loss=6.1267  input='P'       target='arl'
#   Position 10: loss=7.9316  input='arl'     target='oph'
#   Position 11: loss=1.5351  input='oph'     target='ane'
#   Position 12: loss=2.6583  input='ane'     target='),'
#   Position 13: loss=2.3101  input='),'      target=' a'
#   Position 14: loss=9.4594  input=' a'      target=' lively'
#   Position 15: loss=6.1508  input=' lively'  target=' song'
#   Position 16: loss=2.7785  input=' song'   target=' that'
#   Position 17: loss=6.3646  input=' that'   target=' changes'
#   Position 18: loss=5.0784  input=' changes'  target=' tempo'
#   Position 19: loss=15.7533  input=' tempo'  target=' mit'
#   Position 20: loss=12.7555  input=' mit'    target='way'
#   Position 21: loss=4.0621  input='way'     target='.'
#   Position 22: loss=7.1057  input='.'       target=' '
# ======================================================================




