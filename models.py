import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
)
from typing import List, Optional, Dict, Union

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, d_model=80, heads=8, num_layers=1,
                 mlp_dim=160, patch=4):   # ← 正方パッチ4 or 8
        super().__init__()
        self.p = patch

        # LayerNorm用のヘルパークラス
        class ConvLayerNorm2d(nn.Module):
            """Conv2d出力用のLayerNorm (channel-last形式で正規化)"""
            def __init__(self, normalized_shape):
                super().__init__()
                self.norm = nn.LayerNorm(normalized_shape)

            def forward(self, x):
                # x: (B, C, H, W) -> (B, H, W, C)
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                # (B, H, W, C) -> (B, C, H, W)
                x = x.permute(0, 3, 1, 2)
                return x

        self.local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            ConvLayerNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, d_model, 1, bias=False),
            ConvLayerNorm2d(d_model),
            nn.SiLU(inplace=True),
        )

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=mlp_dim,
            dropout=0.0, activation='gelu', batch_first=False, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.fusion = nn.Sequential(
            nn.Conv2d(d_model + in_channels, in_channels, 1, bias=False),
            ConvLayerNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.p
        # 128x1024固定なら常に真。可変入力が来たら早期に落とす。
        assert (H % p == 0) and (W % p == 0), f"H,W must be multiples of {p}"

        y = self.local(x)  # (B, d, H, W)
        B, d, H, W = y.shape
        Hp, Wp = H // p, W // p

        # (B,d,H,W) -> (p*p, B*Hp*Wp, d)
        y = y.view(B, d, Hp, p, Wp, p).permute(3, 5, 0, 2, 4, 1).contiguous()
        y = y.view(p*p, B*Hp*Wp, d)

        y = self.transformer(y)

        # back to (B,d,H,W)
        y = y.view(p, p, B, Hp, Wp, d).permute(2, 5, 3, 0, 4, 1).contiguous()
        y = y.view(B, d, H, W)

        out = torch.cat([x, y], dim=1)
        out = self.fusion(out)
        return out

class HybridBackboneCRNNMobileViT(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(HybridBackboneCRNNMobileViT, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [4, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cntv = 1
        cnt = 1

        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            elif isinstance(m, str) and m.startswith("mobilevit"):
                if m == "mobilevit1":
                    self.features.add_module(f'mvit{cntv}',
                        MobileViTBlock(64,  d_model=80, heads=8, num_layers=1, mlp_dim=160, patch=4))
                elif m == "mobilevit2":
                    self.features.add_module(f'mvit{cntv}',
                        MobileViTBlock(128, d_model=80, heads=8, num_layers=1, mlp_dim=160, patch=8))
                else:
                    raise ValueError(f"unknown mobilevit tag: {m}")
                cntv += 1
            else:
                for i in range(int(m[0])):
                    x = int(m[1])
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce='max'):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCtopC(nn.Module):
    def __init__(self, input_size, nclasses, dropout=0.0):
        super(CTCtopC, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
    
        x = self.dropout(x)

        y = self.cnn_top(x)
        y = y.permute(2, 3, 0, 1)[0]
        return y


class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y



class Connector(nn.Module):
    """Conv1dベースの学習可能なコネクタ（Llama-3.2-3B用）

    改善点:
    - Q-Former (9.5M params) → Conv1d (3.4M params) (64%削減)
    - トークン数: 128 → 21 (学習可能な圧縮)
    - 次元: 512 → 3072 (Linear projection)
    - 重要な情報を学習で保持
    """
    def __init__(self, input_dim=512, output_dim=3072):
        super().__init__()

        # Projection: 512次元 → 3072次元に拡張
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):

        x = self.projection(x)   # (batch, 128, 3072) - 次元拡張
        return x


class LLMWithLLaMA(nn.Module):
    """
    LLaMAモデルのシンプルなラッパークラス
    テキスト生成、ファインチューニング、推論を簡単に実行できる
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B",  # 軽量モデル（3B）
        # model_name: str = "meta-llama/Meta-Llama-3-8B",  # ベースモデル（8B）
    ):
        """
        Args:
            model_name: HuggingFaceのモデル名
        """
        super().__init__()

        print(f"Loading model: {model_name}")

        # LLaMAモデルのロード（CPUでロード、後でnet.to(device)で自動移動）
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # メモリ削減: 16GB→8GB
            low_cpu_mem_usage=True,
        )

        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # モデル情報の取得
        self.config = self.model.config

        print(f"Model loaded successfully!")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Vocab size: {self.config.vocab_size}")
        print(f"   Initial device: CPU (will move to GPU with net.to(device))")

        # LLMパラメータを凍結（学習対象外にする）
        self.model.requires_grad_(False)
        print(f"LLM parameters frozen (8B params not trainable)")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Simplified forward pass (参考コードベース)

        Args:
            inputs_embeds: (batch, seq_len, hidden_size) - RNN出力→Connector変換済み
            labels: (batch, text_len) - テキストのトークンID

        Returns:
            LLM outputs (loss含む)
        """
        # そのまま渡す（参考コードと同じ）
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
        )

        return outputs
        
        
class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru', d_llm=512, enable_connector=True, use_llm=False, use_roberta_aux=False, use_pll_loss=False):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        RNN = nn.GRU if rnn_type == 'gru' else nn.LSTM

        # BiLSTM x3 layers (as per model_structure.md)
        # For LLM path, we need to extract layer1 output, so separate the layers
        self.rec1 = RNN(input_size, hidden, num_layers=1, bidirectional=True, dropout=0.0)

        self.recN = None
        if num_layers > 1:
            self.recN = RNN(2*hidden, hidden, num_layers=num_layers-1, bidirectional=True, dropout=.2)

        # Final CTC projection (for BiLSTM layer3 final output)
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        # BiLSTM layer1用の専用CTC projection
        self.fnl_layer1 = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5),
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

        # LLM使用時のみ Connector と LLM をロード
        self.use_llm = use_llm
        if use_llm:
            print("Loading LLM components (Connector + LLaMA-3.2-3B)...")
            self.connector = Connector(input_dim=512)
            self.llm = LLMWithLLaMA()
        else:
            print("LLM disabled: Using CNN shortcut only")
            self.connector = None
            self.llm = None

        # RoBERTa補助損失用のコンポーネント
        self.use_roberta_aux = use_roberta_aux
        if use_roberta_aux:
            print("Loading RoBERTa auxiliary loss components...")
            # Projection層: CTC確率分布(nclasses次元) → RoBERTa入力(768次元)
            self.projection_roberta = nn.Linear(nclasses, 768)

            # RoBERTaモデル（凍結）
            from transformers import RobertaForMaskedLM, RobertaTokenizer
            self.roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
            self.roberta.requires_grad_(False)  # RoBERTaは凍結、projectionのみ学習

            # トークナイザー
            self.tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
            print("RoBERTa auxiliary loss loaded (roberta-base, params frozen)")
        else:
            self.projection_roberta = None
            self.roberta = None
            self.tokenizer_roberta = None

        # RoBERTa PLL損失用のコンポーネント
        self.use_pll_loss = use_pll_loss
        self.nclasses = nclasses
        if use_pll_loss:
            print("Loading RoBERTa PLL loss components...")
            # RoBERTaモデルとトークナイザー（凍結）
            from transformers import RobertaForMaskedLM, RobertaTokenizer
            if not use_roberta_aux:
                # roberta_auxが無効の場合、ここで新たにロード
                self.roberta_pll = RobertaForMaskedLM.from_pretrained("roberta-base")
                self.roberta_pll.requires_grad_(False)
                self.tokenizer_pll = RobertaTokenizer.from_pretrained("roberta-base")
            else:
                # roberta_auxが有効な場合、同じモデルを共有
                self.roberta_pll = self.roberta
                self.tokenizer_pll = self.tokenizer_roberta
            print("RoBERTa PLL loss loaded (roberta-base, params frozen)")
        else:
            self.roberta_pll = None
            self.tokenizer_pll = None

    def ctc_decode_batch(self, ctc_logits, classes):
        """
        CTCロジットをバッチでデコードして文字列に変換

        Args:
            ctc_logits: (width, batch, nclasses) CTCロジット
            classes: クラス名のリスト

        Returns:
            List[str]: デコードされた文字列のリスト
        """
        batch_size = ctc_logits.size(1)
        predicted_indices = torch.argmax(ctc_logits, dim=2)  # (width, batch)

        decoded_texts = []
        for b in range(batch_size):
            indices = predicted_indices[:, b]  # (width,)
            decoded = []
            prev_idx = -1
            for idx in indices:
                idx = idx.item()
                if idx != 0 and idx != prev_idx:  # 0はブランク、連続文字はスキップ
                    if idx - 1 < len(classes):
                        decoded.append(classes[idx - 1])
                prev_idx = idx
            decoded_texts.append(''.join(decoded))

        return decoded_texts

    def compute_pll_loss(self, pred_texts, label_texts, device):
        """
        Pseudo Log-Likelihood (PLL) に基づく損失を計算

        Args:
            pred_texts: List[str] 予測文字列
            label_texts: List[str] 正解文字列
            device: デバイス

        Returns:
            torch.Tensor: PLL損失 (スカラー)
        """
        import torch.nn.functional as F

        total_loss = 0.0
        valid_count = 0

        for pred_text, label_text in zip(pred_texts, label_texts):
            # 空文字列の場合はスキップ
            if len(pred_text) == 0 or len(label_text) == 0:
                continue

            # 予測文字列のPLLを計算
            pll_pred = self._compute_pll_score(pred_text, device)

            # 正解文字列のPLLを計算
            pll_label = self._compute_pll_score(label_text, device)

            # 差分損失: PLL(label) - PLL(pred)
            # 負の値を避けるためReLU適用
            loss = torch.relu(torch.tensor(pll_label - pll_pred, device=device))
            total_loss += loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_count

    def _compute_pll_score(self, sentence, device):
        """
        文字列のPseudo Log-Likelihood (PLL) を計算

        Args:
            sentence: str 入力文字列
            device: デバイス

        Returns:
            float: 平均対数尤度
        """
        import torch.nn.functional as F

        # トークン化
        enc = self.tokenizer_pll(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        seq_len = input_ids.size(1)

        # マスク位置（special tokensを除く）
        mask_positions = list(range(1, seq_len - 1))
        if len(mask_positions) == 0:
            return float("-inf")

        total_log_prob = 0.0
        n = 0

        with torch.no_grad():
            for pos in mask_positions:
                # 位置posをマスク
                masked = input_ids.clone()
                masked[0, pos] = self.tokenizer_pll.mask_token_id

                # RoBERTaで予測
                outputs = self.roberta_pll(input_ids=masked, attention_mask=attention_mask)
                logits = outputs.logits  # (1, seq_len, vocab)

                # マスク位置の対数確率
                log_probs = F.log_softmax(logits[0, pos], dim=-1)
                true_id = input_ids[0, pos].item()
                lp = log_probs[true_id].item()
                total_log_prob += lp
                n += 1

        avg_log_prob = total_log_prob / n
        return avg_log_prob


    def forward(self, x, y_llm=None, transcr_llm=None, y_roberta=None, transcr_roberta=None, y_pll=None, transcr_pll=None, classes=None):
        """
        Args:
            x: 全サンプルの特徴量 (batch_size, 256, 1, width)
            y_llm: LLM用サンプルの特徴量 (llm_batch_size, 256, 1, width)
            transcr_llm: LLM用の正解文字列 (llm_batch_size,)
            y_roberta: RoBERTa用サンプルの特徴量 (roberta_batch_size, 256, 1, width)
            transcr_roberta: RoBERTa用の正解文字列 (roberta_batch_size,)
            y_pll: PLL損失用サンプルの特徴量 (pll_batch_size, 256, 1, width)
            transcr_pll: PLL損失用の正解文字列 (pll_batch_size,)
            classes: クラス名のリスト（CTCデコード用）
        """
        # RNN処理（全サンプル）
        y = x.permute(2, 3, 0, 1)[0]  # (width, batch, 256)
        y1 = self.rec1(y)[0]  # (width, batch, 512) - BiLSTM layer1 output

        # Pass through remaining layers
        if self.recN is not None:
            y_rnn = self.recN(y1)[0]  # (width, batch, 512) - BiLSTM layers 2-3 output
        else:
            y_rnn = y1

        # Final CTC projection
        y_ctc = self.fnl(y_rnn)  # (width, batch, nclasses)

        # LLM処理（use_llm=true かつ 選択されたサンプルのみ）
        output_llm = None
        if self.use_llm and y_llm is not None and transcr_llm is not None and self.training:
            # y_llmからRNN layer1の出力を取得（as per model_structure.md）
            y_llm_seq = y_llm.permute(2, 3, 0, 1)[0]  # (width, llm_batch, 256)
            y1_llm = self.rec1(y_llm_seq)[0]  # (width, llm_batch, 512) - layer1 output only

            # Connectorで3072次元に変換 (Llama-3.2-3B用)
            prefix_input = y1_llm.permute(1, 0, 2)  # (llm_batch, width, 512)

            # 🔍 デバッグ: 形状確認
            # print(f"\n{'='*60}")
            # print(f"[DEBUG] Shape verification")
            # print(f"{'='*60}")
            # print(f"y1_llm.shape:       {y1_llm.shape} (width, llm_batch, 512)")
            # print(f"prefix_input.shape: {prefix_input.shape} (llm_batch, width, 512)")
            # print(f"Expected:           (llm_batch, 128, 512)")

            inputs_embeds = self.connector(prefix_input)   # (llm_batch,128, 3072)

            # print(f"inputs_embeds.shape: {inputs_embeds.shape}")
            # print(f"Expected:            (llm_batch, 21, 3072)")


            llm_labels = self.llm.tokenizer(
                list(transcr_llm),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=inputs_embeds.shape[1]  # Connector出力の長さに合わせる
            )
            labels = llm_labels["input_ids"].to(y_llm.device)  # (llm_batch, 128)

            # print(f"labels.shape:        {labels.shape}")
            # print(f"Expected:            (llm_batch, 128)")
            # print(f"{'='*60}\n")


            output_llm = self.llm(
                inputs_embeds=inputs_embeds.half(),  # (batch, 128, 3072) float16に変換
                labels=labels                         # (batch, 128) ← 長さ一致！
            )

        # RoBERTa補助損失の計算（use_roberta_aux=true かつ 選択されたサンプルのみ）
        output_roberta = None
        if self.use_roberta_aux and y_roberta is not None and transcr_roberta is not None and self.training:
            # # y_robertaからRNN layer1の出力を取得
            # y_roberta_seq = y_roberta.permute(2, 3, 0, 1)[0]  # (width, roberta_batch, 256)
            # y1_roberta = self.rec1(y_roberta_seq)[0]  # (width, roberta_batch, 512) - layer1 output

            # # Projectionで768次元に変換 (RoBERTa用)
            # roberta_input = y1_roberta.permute(1, 0, 2)  # (roberta_batch, width, 512)
            # roberta_embeds = self.projection_roberta(roberta_input)  # (roberta_batch, width, 768)
            
            
            # y_robertaからRNN layer1の出力を取得
            y_roberta_seq = y_roberta.permute(2, 3, 0, 1)[0]  # (width, roberta_batch, 256)
            y1_roberta = self.rec1(y_roberta_seq)[0]  # (width, roberta_batch, 512) - BiLSTM layer1 output

            # CTCロジット (単語確率) を計算
            ctc_logits = self.fnl(y1_roberta)  # (width, roberta_batch, nclasses)
            # ソフトマックスで確率分布に変換
            ctc_probs = F.softmax(ctc_logits, dim=-1)  # (width, roberta_batch, nclasses)
            # 確率分布をRoBERTa用の埋め込み空間に射影 (nclasses -> 768次元)
            roberta_embeds = self.projection_roberta(
                ctc_probs.permute(1, 0, 2)  # (roberta_batch, width, nclasses)
            )  # (roberta_batch, width, 768)

            # 正解テキストをトークン化
            roberta_labels = self.tokenizer_roberta(
                list(transcr_roberta),
                return_tensors="pt",
                padding="max_length",
                max_length=roberta_embeds.shape[1],  # widthに合わせる
                truncation=True
            ).input_ids.to(y_roberta.device)

            # RoBERTa損失を計算
            output_roberta = self.roberta(inputs_embeds=roberta_embeds, labels=roberta_labels)

        # RoBERTa PLL損失の計算（use_pll_loss=true かつ 選択されたサンプルのみ）
        pll_loss_bilstm = None
        pll_loss_mobilevit = None

        if self.use_pll_loss and y_pll is not None and transcr_pll is not None and classes is not None and self.training:
            # BiLSTM layer1 PLL損失
            # y_pllからRNN layer1の出力を取得
            y_pll_seq = y_pll.permute(2, 3, 0, 1)[0]  # (width, pll_batch, 256)
            y1_pll = self.rec1(y_pll_seq)[0]  # (width, pll_batch, 512) - BiLSTM layer1 output

            # CTCロジットを計算
            ctc_logits_pll_bilstm = self.fnl(y1_pll)  # (width, pll_batch, nclasses)

            # CTCデコード
            pred_texts_bilstm = self.ctc_decode_batch(ctc_logits_pll_bilstm, classes)

            # BiLSTM layer1のPLL損失を計算
            pll_loss_bilstm = self.compute_pll_loss(pred_texts_bilstm, list(transcr_pll), y_pll.device)

            # MobileViT PLL損失
            # MobileViT出力から直接CTCロジットを生成（CNN shortcut使用）
            mobilevit_ctc = self.cnn(y_pll)  # (pll_batch, nclasses, 1, width)
            mobilevit_ctc = mobilevit_ctc.permute(2, 3, 0, 1)[0]  # (width, pll_batch, nclasses)

            # CTCデコード
            pred_texts_mobilevit = self.ctc_decode_batch(mobilevit_ctc, classes)

            # MobileViTのPLL損失を計算
            pll_loss_mobilevit = self.compute_pll_loss(pred_texts_mobilevit, list(transcr_pll), y_pll.device)

        if self.training:
            # BiLSTM layer1のCTC出力も計算（学習用）
            y1_ctc_train = self.fnl_layer1(y1)  # (width, batch, nclasses)
            return y_ctc, self.cnn(x).permute(2, 3, 0, 1)[0], y1_ctc_train, output_llm, output_roberta, pll_loss_bilstm, pll_loss_mobilevit
        else:
            # 推論時もBiLSTM layer1出力を計算して返す（sample_decoding用）
            y_seq_infer = x.permute(2, 3, 0, 1)[0]  # (width, batch, 256)
            y1_infer = self.rec1(y_seq_infer)[0]  # BiLSTM layer1出力 (width, batch, 512)
            y1_ctc_infer = self.fnl_layer1(y1_infer)  # layer1専用のCTC projection使用

            return y_ctc, self.cnn(x).permute(2, 3, 0, 1)[0], y1_ctc_infer  # 3つ返す


class HTRNet(nn.Module):
    def __init__(self, arch_cfg, nclasses, use_llm=False, use_roberta_aux=False, use_pll_loss=False):
        super(HTRNet, self).__init__()

        if arch_cfg.stn:
            raise NotImplementedError('Spatial Transformer Networks not implemented - you can easily build your own!')
            #self.stn = STN()
        else:
            self.stn = None

        cnn_cfg = arch_cfg.cnn_cfg
        self.features = HybridBackboneCRNNMobileViT(arch_cfg.cnn_cfg, flattening=arch_cfg.flattening)

        if arch_cfg.flattening=='maxpool' or arch_cfg.flattening=='avgpool':
            hidden = cnn_cfg[-1][-1]
        elif arch_cfg.flattening=='concat':
            hidden = 2 * 8 * cnn_cfg[-1][-1]
        else:
            print('problem! - no such flattening is defined')

        head = arch_cfg.head_type
        if head=='cnn':
            self.top = CTCtopC(hidden, nclasses)
        elif head=='rnn':
            self.top = CTCtopR(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)
        elif head=='both':
            self.top = CTCtopB(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type, use_llm=use_llm, use_roberta_aux=use_roberta_aux, use_pll_loss=use_pll_loss)

        # LM損失用の射影層（BiLSTM hidden → LM embedding）
        # BiLSTMはbidirectionalなので、出力は 2 * rnn_hidden_size
        if head=='both':
            bilstm_output_dim = 2 * arch_cfg.rnn_hidden_size  # 2 * 256 = 512
        else:
            bilstm_output_dim = 2 * arch_cfg.rnn_hidden_size if head=='rnn' else hidden
        self.hidden_to_lm_proj = nn.Linear(bilstm_output_dim, 768)

    def forward(self, x, img_llm=None, transcr_llm=None, img_roberta=None, transcr_roberta=None, img_pll=None, transcr_pll=None, classes=None):
        """
        Args:
            x: 全サンプルの画像 (batch_size, C, H, W)
            img_llm: LLM用サンプルの画像 (llm_batch_size, C, H, W)
            transcr_llm: LLM用の正解文字列 (llm_batch_size,)
            img_roberta: RoBERTa用サンプルの画像 (roberta_batch_size, C, H, W)
            transcr_roberta: RoBERTa用の正解文字列 (roberta_batch_size,)
            img_pll: PLL損失用サンプルの画像 (pll_batch_size, C, H, W)
            transcr_pll: PLL損失用の正解文字列 (pll_batch_size,)
            classes: クラス名のリスト（CTCデコード用）
        """
        # 全サンプルの特徴量抽出
        if self.stn is not None:
            x = self.stn(x)
        y = self.features(x)

        # LLM用サンプルの特徴量抽出
        y_llm = None
        if img_llm is not None:
            if self.stn is not None:
                img_llm = self.stn(img_llm)
            y_llm = self.features(img_llm)

        # RoBERTa用サンプルの特徴量抽出
        y_roberta = None
        if img_roberta is not None:
            if self.stn is not None:
                img_roberta = self.stn(img_roberta)
            y_roberta = self.features(img_roberta)

        # PLL損失用サンプルの特徴量抽出
        y_pll = None
        if img_pll is not None:
            if self.stn is not None:
                img_pll = self.stn(img_pll)
            y_pll = self.features(img_pll)

        # CTCtopBに渡す
        if transcr_llm is not None or transcr_roberta is not None or transcr_pll is not None:
            y = self.top(y, y_llm=y_llm, transcr_llm=transcr_llm, y_roberta=y_roberta, transcr_roberta=transcr_roberta, y_pll=y_pll, transcr_pll=transcr_pll, classes=classes)
        else:
            y = self.top(y)

        return y


# =============================================================================
# Language Model Loss Functions
# =============================================================================

def calculate_lm_loss_single(text, lm_model, tokenizer, device):
    """
    単一テキストのLanguage Model lossを計算

    Args:
        text: 入力テキスト
        lm_model: 言語モデル (GPT-2など)
        tokenizer: トークナイザ
        device: デバイス (cuda/cpu)

    Returns:
        float: Cross Entropy Loss（エラー時はinf）
    """
    if len(text) == 0:
        return float('inf')

    try:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        if input_ids.size(1) < 2:  # トークンが1つ以下の場合はロス計算不可
            return torch.tensor(float('inf'), device=device)

        # with torch.no_grad(): を削除（勾配を流すため）
        outputs = lm_model(input_ids)
        logits = outputs.logits

        # 次トークン予測のためにシフト
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Cross Entropy Loss計算
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return ce_loss  # Tensorのまま返す（.item()しない）

    except Exception as e:
        # トークナイゼーションエラーなどの場合
        return float('inf')


def calculate_lm_loss_diff(pred_text, label_text, lm_model, tokenizer, device):
    """
    CTC予測と正解文字列のLM loss差を計算

    Args:
        pred_text: CTC予測文字列
        label_text: 正解文字列
        lm_model: 言語モデル
        tokenizer: トークナイザ
        device: デバイス

    Returns:
        torch.Tensor: max(0, pred_loss - label_loss) （予測が正解より悪い場合のみペナルティ）
    """
    pred_loss = calculate_lm_loss_single(pred_text, lm_model, tokenizer, device)
    label_loss = calculate_lm_loss_single(label_text, lm_model, tokenizer, device)

    # 両方とも有効な場合のみ差分を計算
    # infチェック（Tensorの場合）
    if torch.isinf(pred_loss) or torch.isinf(label_loss):
        return torch.tensor(0.0, device=device)

    # 予測が正解より悪い場合のみペナルティ
    diff = pred_loss - label_loss
    return torch.clamp(diff, min=0.0)  # max(0, diff) の微分可能版


def calculate_lm_loss_batch(pred_texts, label_texts, lm_model, tokenizer, device):
    """
    バッチ全体のLM loss差の平均を計算

    Args:
        pred_texts: CTC予測文字列のリスト
        label_texts: 正解文字列のリスト
        lm_model: 言語モデル
        tokenizer: トークナイザ
        device: デバイス

    Returns:
        torch.Tensor: バッチ平均LM loss差（微分可能）
    """
    total_loss = torch.tensor(0.0, device=device)  # Tensorで初期化
    valid_count = 0

    for pred, label in zip(pred_texts, label_texts):
        diff = calculate_lm_loss_diff(pred, label, lm_model, tokenizer, device)
        if diff.item() > 0:  # Tensorの値をチェック
            total_loss = total_loss + diff  # Tensor加算（勾配グラフを保持）
            valid_count += 1

    # 有効なサンプルがない場合は0を返す
    if valid_count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / valid_count  # Tensorのまま返す


def calculate_lm_loss_from_hidden_states(
    hidden_states, true_texts, lm_model, tokenizer, projection, device
):
    """
    BiLSTM中間層の隠れ状態から微分可能なLM損失を計算（argmax不使用）

    Args:
        hidden_states: (width, batch, hidden_dim) - BiLSTM layer1出力
        true_texts: 正解テキストのリスト
        lm_model: GPT-2モデル
        tokenizer: GPT-2トークナイザ
        projection: nn.Linear(256, 768) - 学習可能な射影層
        device: デバイス

    Returns:
        torch.Tensor: 微分可能なLM損失（スカラー、grad_fn付き）
    """
    import torch.nn.functional as F

    # 射影: BiLSTM hidden(256) → GPT-2 embed(768)
    projected = projection(hidden_states)  # (width, batch, 768)

    # プーリングで長さ圧縮: 128 → 32
    pooled = F.avg_pool1d(
        projected.permute(1, 2, 0),  # (batch, 768, width)
        kernel_size=4, stride=4
    ).permute(2, 0, 1)  # (32, batch, 768)

    # LM入力形式に変換
    lm_input = pooled.permute(1, 0, 2)  # (batch, 32, 768)

    # 正解テキストをトークン化
    encodings = tokenizer(
        true_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    labels = encodings["input_ids"].to(device)

    # LMに入力（微分可能！）
    outputs = lm_model(inputs_embeds=lm_input, labels=labels)

    # 損失を返す（Tensorのまま）
    return outputs.loss