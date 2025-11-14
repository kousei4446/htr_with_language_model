import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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

    def forward(self, x, reduce='max', return_mobilevit_outputs=False):

        y = x
        mobilevit_outputs = []

        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

            # MobileViTBlockの出力を保存
            if isinstance(nn_module, MobileViTBlock):
                mobilevit_outputs.append(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        if return_mobilevit_outputs:
            return y, mobilevit_outputs
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



class ConnectorForMobileViT(nn.Module):
    """MobileViT出力用のコネクタ (GPT-2 small用、Q-Formerベース)

    MobileViTの2D特徴マップ (H, W, C) をGPT-2の入力形式 (seq_len, hidden_size=768) に変換
    テキスト埋め込みをクエリとして使い、画像特徴から関連情報を抽出
    """
    def __init__(self, input_channels, output_dim=768, num_heads=8, ffn_ratio=2):
        """
        Args:
            input_channels: MobileViT出力のチャネル数 (64 or 128)
            output_dim: GPT-2のhidden_size (768)
            num_heads: Q-Formerのアテンションヘッド数 (8)
            ffn_ratio: Q-FormerのFFN次元倍率 (2)
        """
        super().__init__()

        # Spatial pooling: (B, C, H, W) → (B, C, 1, W)
        # Height方向をmax poolingで圧縮
        self.spatial_pool = nn.AdaptiveMaxPool2d((1, None))

        # Q-Former: テキスト埋め込みクエリで画像特徴から情報抽出
        self.qformer = LightweightQFormerConnector(
            input_dim=input_channels,
            output_dim=output_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio
        )

    def forward(self, x, text_embeddings):
        """
        Args:
            x: (batch, channels, height, width) - MobileViT出力
            text_embeddings: (batch, text_len, 768) - GPT-2埋め込みから得たテキスト表現
        Returns:
            (batch, text_len, 768) - 画像情報で強化されたテキスト表現
        """
        # (B, C, H, W) → (B, C, 1, W)
        x = self.spatial_pool(x)

        # (B, C, 1, W) → (B, W, C)
        x = x.squeeze(2).permute(0, 2, 1)

        # Q-Formerで画像特徴から情報抽出
        # (B, W, C) + (B, text_len, 768) → (B, text_len, 768)
        x = self.qformer(x, text_embeddings)

        return x


class ConnectorForBiLSTM(nn.Module):
    """BiLSTM出力用のコネクタ (GPT-2 small用、Q-Formerベース)

    BiLSTMの出力 (seq_len, hidden_size) をGPT-2の入力形式に変換
    テキスト埋め込みをクエリとして使い、BiLSTM特徴から関連情報を抽出
    """
    def __init__(self, input_dim=512, output_dim=768, num_heads=8, ffn_ratio=2):
        """
        Args:
            input_dim: BiLSTM出力の次元 (512: bidirectional 256*2)
            output_dim: GPT-2のhidden_size (768)
            num_heads: Q-Formerのアテンションヘッド数 (8)
            ffn_ratio: Q-FormerのFFN次元倍率 (2)
        """
        super().__init__()

        # Q-Former: テキスト埋め込みクエリでBiLSTM特徴から情報抽出
        self.qformer = LightweightQFormerConnector(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio
        )

    def forward(self, x, text_embeddings):
        """
        Args:
            x: (batch, seq_len, 512) - BiLSTM出力
            text_embeddings: (batch, text_len, 768) - GPT-2埋め込みから得たテキスト表現
        Returns:
            (batch, text_len, 768) - BiLSTM情報で強化されたテキスト表現
        """
        return self.qformer(x, text_embeddings)


class LightweightQFormerConnector(nn.Module):
    """超軽量Q-Formerコネクタ（テキスト埋め込みクエリ方式）

    GPT-2の埋め込み層から得たテキスト表現をクエリとして使用し、
    Cross-Attentionで画像特徴から関連情報を選択的に抽出する。

    構成:
    - Self-Attention削除（テキスト間の関係はGPT-2が学習）
    - 1層のCross-Attention + FFN
    - FFN次元: 768*2（軽量化）
    - パラメータ数: 約4.8M/コネクタ
    """
    def __init__(self, input_dim, output_dim=768, num_heads=8, ffn_ratio=2):
        """
        Args:
            input_dim: 入力特徴の次元（MobileViT: 64/128, BiLSTM: 512）
            output_dim: GPT-2のhidden_size (768)
            num_heads: アテンションヘッド数 (8)
            ffn_ratio: FFN次元倍率 (2 → 768*2=1536)
        """
        super().__init__()

        # 画像特徴投影: input_dim → 768
        self.input_projection = nn.Linear(input_dim, output_dim)

        # テキスト埋め込み投影 (クエリ生成用)
        self.text_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Cross-Attention (1層のみ)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(output_dim)

        # FFN (768 → 1536 → 768)
        ffn_dim = output_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, output_dim)
        )
        self.ffn_norm = nn.LayerNorm(output_dim)

    def forward(self, x, text_embeddings):
        """
        Args:
            x: (batch, seq_len, input_dim) - 画像特徴（コネクタ出力）
            text_embeddings: (batch, text_len, 768) - GPT-2埋め込みから得たテキスト表現
        Returns:
            (batch, text_len, 768) - 画像情報で強化されたテキスト表現
        """
        # 画像特徴を投影
        x = self.input_projection(x)  # (batch, seq_len, 768)

        # テキスト埋め込みをクエリに変換（dtype変換）
        text_embeddings = text_embeddings.to(dtype=self.text_projection[0].weight.dtype, device=text_embeddings.device)
        queries = self.text_projection(text_embeddings)  # (batch, text_len, 768)

        # Cross-Attention: テキストクエリで画像特徴から情報抽出
        attn_out, _ = self.cross_attn(queries, x, x)  # (batch, text_len, 768)
        queries = self.attn_norm(queries + attn_out)  # Residual connection

        # FFN
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)  # Residual connection

        return queries


class LLMWithGPT2(nn.Module):
    """
    GPT-2 smallモデルのシンプルなラッパークラス
    テキスト生成、ファインチューニング、推論を簡単に実行できる
    """
    def __init__(
        self,
        model_name: str = "gpt2",  # GPT-2 small (124M params, hidden_size=768)
    ):
        """
        Args:
            model_name: HuggingFaceのモデル名
        """
        super().__init__()

        print(f"[*] Loading model: {model_name}")

        # GPT-2モデルのロード（CPUでロード、後でnet.to(device)で自動移動）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # メモリ削減
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

        print(f"[+] Model loaded successfully!")
        print(f"    Hidden size: {self.config.hidden_size}")
        print(f"    Vocab size: {self.config.vocab_size}")
        print(f"    Initial device: CPU (will move to GPU with net.to(device))")

        # LLMパラメータを凍結（学習対象外にする）
        self.model.requires_grad_(False)
        print(f"[!] LLM parameters frozen (124M params not trainable)")

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
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru', d_llm=512, enable_connector=True, use_llm=False):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        RNN = nn.GRU if rnn_type == 'gru' else nn.LSTM

        # BiLSTM x3 layers (as per model_structure.md)
        # For LLM path, we need to extract layer1 output, so separate the layers
        self.rec1 = RNN(input_size, hidden, num_layers=1, bidirectional=True, dropout=0.0)

        self.recN = None
        if num_layers > 1:
            self.recN = RNN(2*hidden, hidden, num_layers=num_layers-1, bidirectional=True, dropout=.2)

        # Final CTC projection
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5),
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

        # LLM使用時のみ Connector と LLM をロード
        self.use_llm = use_llm
        if use_llm:
            print("[*] Loading LLM components (Connectors + GPT-2 small)...")
            # 3つのConnector: MobileViT1 (64ch), MobileViT2 (128ch), BiLSTM layer1 (512)
            self.connector_mvit1 = ConnectorForMobileViT(input_channels=64, output_dim=768)
            self.connector_mvit2 = ConnectorForMobileViT(input_channels=128, output_dim=768)
            self.connector_bilstm = ConnectorForBiLSTM(input_dim=512, output_dim=768)
            self.llm = LLMWithGPT2()
        else:
            print("[!] LLM disabled: Using CNN shortcut only")
            self.connector_mvit1 = None
            self.connector_mvit2 = None
            self.connector_bilstm = None
            self.llm = None
        
        
    def forward(self, x, mobilevit_outputs=None, transcr_llm=None):
        """
        Args:
            x: 全サンプルの特徴量 (batch_size, 256, 1, width)
            mobilevit_outputs: MobileViTの中間出力リスト [mvit1_output, mvit2_output]
                - mvit1_output: (batch, 64, H, W) またはLLM用の場合 (llm_batch, 64, H, W)
                - mvit2_output: (batch, 128, H, W) またはLLM用の場合 (llm_batch, 128, H, W)
            transcr_llm: LLM用の正解文字列 (llm_batch_size,)
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

        # LLM処理（use_llm=true かつ 学習時のみ）
        llm_outputs = {}
        if self.use_llm and mobilevit_outputs is not None and transcr_llm is not None and self.training:
            # 3つのパスで次トークン予測損失を計算
            mvit1_output, mvit2_output = mobilevit_outputs

            # 正解文字列をトークン化
            llm_labels = self.llm.tokenizer(
                list(transcr_llm),
                return_tensors="pt",
                padding=True,  # バッチ内で長さを揃える
                truncation=True,
                max_length=512  # GPT-2の最大コンテキスト長を考慮
            )
            labels = llm_labels["input_ids"].to(x.device)  # (llm_batch, text_len)

            # GPT-2の埋め込み層でテキスト埋め込みを取得
            with torch.no_grad():  # クエリとして使うだけなので勾配不要
                text_embeddings = self.llm.model.transformer.wte(labels)  # (llm_batch, text_len, 768)

            # 各コネクタにテキスト埋め込みを渡す
            # Q-Formerが画像特徴からテキストに関連する情報を抽出

            # 1. MobileViT1出力 → Q-Former → GPT-2入力形式
            inputs_embeds_mvit1 = self.connector_mvit1(mvit1_output, text_embeddings)  # (llm_batch, text_len, 768)

            # 2. MobileViT2出力 → Q-Former → GPT-2入力形式
            inputs_embeds_mvit2 = self.connector_mvit2(mvit2_output, text_embeddings)  # (llm_batch, text_len, 768)

            # 3. BiLSTM layer1出力 → Q-Former → GPT-2入力形式
            y1_llm = y1[:, :text_embeddings.size(0), :]  # (width, llm_batch, 512) - LLMサンプルのみ抽出
            inputs_embeds_bilstm = self.connector_bilstm(y1_llm.permute(1, 0, 2), text_embeddings)  # (llm_batch, text_len, 768)

            # すべて (llm_batch, text_len, 768) で統一されている
            # 最小長への切り詰め処理は不要（Q-Formerが自動的に調整）

            # 各パスでGPT-2を実行
            llm_outputs['mobilevit1'] = self.llm(
                inputs_embeds=inputs_embeds_mvit1.half(),
                labels=labels
            )

            llm_outputs['mobilevit2'] = self.llm(
                inputs_embeds=inputs_embeds_mvit2.half(),
                labels=labels
            )

            llm_outputs['bilstm_layer1'] = self.llm(
                inputs_embeds=inputs_embeds_bilstm.half(),
                labels=labels
            )

        if self.training:
            return y_ctc, self.cnn(x).permute(2, 3, 0, 1)[0], llm_outputs
        else:
            return y_ctc, self.cnn(x).permute(2, 3, 0, 1)[0]


class HTRNet(nn.Module):
    def __init__(self, arch_cfg, nclasses, use_llm=False):
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
            self.top = CTCtopB(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type, use_llm=use_llm)

    def forward(self, x, img_llm=None, transcr_llm=None):
        """
        Args:
            x: 全サンプルの画像 (batch_size, C, H, W)
            img_llm: LLM用サンプルの画像 (llm_batch_size, C, H, W)
            transcr_llm: LLM用の正解文字列 (llm_batch_size,)
        """
        # 全サンプルの特徴量抽出
        if self.stn is not None:
            x = self.stn(x)

        # LLM用の場合はMobileViTの中間出力も取得
        mobilevit_outputs_llm = None
        if img_llm is not None and self.top.use_llm:
            if self.stn is not None:
                img_llm = self.stn(img_llm)
            y_llm, mobilevit_outputs_llm = self.features(img_llm, return_mobilevit_outputs=True)

        # 通常の特徴量抽出（MobileViT中間出力は不要）
        y = self.features(x, return_mobilevit_outputs=False)

        # CTCtopBに渡す
        if transcr_llm is not None and mobilevit_outputs_llm is not None:
            y = self.top(y, mobilevit_outputs=mobilevit_outputs_llm, transcr_llm=transcr_llm)
        else:
            y = self.top(y)

        return y

    def freeze_except_connectors(self):
        """Freeze all parameters except connector layers"""
        # Freeze CNN backbone
        for param in self.features.parameters():
            param.requires_grad = False

        # Freeze BiLSTM layers
        if hasattr(self.top, 'rec1'):
            for param in self.top.rec1.parameters():
                param.requires_grad = False
        if hasattr(self.top, 'recN'):
            for param in self.top.recN.parameters():
                param.requires_grad = False

        # Freeze CTC heads
        if hasattr(self.top, 'fnl'):
            for param in self.top.fnl.parameters():
                param.requires_grad = False
        if hasattr(self.top, 'cnn'):
            for param in self.top.cnn.parameters():
                param.requires_grad = False

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Frozen all parameters except connectors")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def freeze_connector(self):
        """Freeze ONLY connector layers"""
        if not hasattr(self.top, 'use_llm') or not self.top.use_llm:
            print("LLM is not used; no connectors to freeze.")
            return

        # Freeze connector layers with existence check
        if hasattr(self.top, 'connector_mvit1') and self.top.connector_mvit1 is not None:
            for param in self.top.connector_mvit1.parameters():
                param.requires_grad = False

        if hasattr(self.top, 'connector_mvit2') and self.top.connector_mvit2 is not None:
            for param in self.top.connector_mvit2.parameters():
                param.requires_grad = False

        if hasattr(self.top, 'connector_bilstm') and self.top.connector_bilstm is not None:
            for param in self.top.connector_bilstm.parameters():
                param.requires_grad = False

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Frozen connector parameters")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        