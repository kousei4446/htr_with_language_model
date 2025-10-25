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

        self.local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=mlp_dim,
            dropout=0.0, activation='gelu', batch_first=False, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.fusion = nn.Sequential(
            nn.Conv2d(d_model + in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
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


class QFormer(nn.Module):
    """BLIP-2スタイルのQ-Former: 128トークン→64トークンに圧縮"""
    def __init__(self, input_dim=512, num_queries=64, num_heads=8):
        super().__init__()

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, input_dim))

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch, 128, 512)
        batch_size = x.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        # (batch, 64, 512)

        # Cross-attention: queries attend to x
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = queries + attn_out  # Residual connection

        # Feed-forward
        queries = queries + self.ffn(self.norm(queries))  # Residual connection

        return queries  # (batch, 64, 512)


class Connector(nn.Module):
    """Q-Former + 2段階拡張（バランス型）

    パラメータ数: 約7.92M (従来11.16Mから29%削減)
    トークン数: 128 → 64 (50%削減)
    """
    def __init__(self, input_dim=512, num_queries=64):
        super().__init__()

        # Q-Former: 128トークン → 64トークンに圧縮
        self.qformer = QFormer(
            input_dim=input_dim,
            num_queries=num_queries,
            num_heads=8
        )

        # 2段階拡張: 512 → 1024 → 4096
        self.expansion = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 4096),
            nn.LayerNorm(4096),
        )

    def forward(self, x):
        # x: (batch, 128, 512)
        x = self.qformer(x)      # (batch, 64, 512)
        x = self.expansion(x)    # (batch, 64, 4096)
        return x


class LLMWithLLaMA(nn.Module):
    """
    LLaMAモデルのシンプルなラッパークラス
    テキスト生成、ファインチューニング、推論を簡単に実行できる
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",  # ベースモデル（推奨）
        # model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",  # Instructモデル
        device: str = "cpu",  # デフォルトCPUでロード→後でnet.to(device)で移動
    ):
        """
        Args:
            model_name: HuggingFaceのモデル名
            device: 使用するデバイス（'cuda'または'cpu'）
        """
        super().__init__()

        print(f"📦 Loading model: {model_name}")

        # LLaMAモデルのロード（fp16でメモリ削減）
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # メモリ削減: 16GB→8GB
            low_cpu_mem_usage=True,
        ).to(device)
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # モデル情報の取得
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        
        print(f"✅ Model loaded successfully!")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Vocab size: {self.config.vocab_size}")
        print(f"   Device: {self.device}")

        # LLMパラメータを凍結（学習対象外にする）
        self.model.requires_grad_(False)
        print(f"🔒 LLM parameters frozen (8B params not trainable)")

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
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru',d_llm=512, enable_connector=True):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg
        
        RNN = nn.GRU if rnn_type == 'gru' else nn.LSTM
        
        self.rec1 = RNN(input_size, hidden, num_layers=1, bidirectional=True, dropout=0.0)

        # Bidirectional RNN出力を統合する層
        self.rnn_projection = nn.Sequential(
            nn.LayerNorm(2 * hidden),  # 512次元を正規化
            nn.Linear(2 * hidden, 2 * hidden),  # 512→512
            nn.GELU(),
        )

        self.recN = None
        if num_layers > 1:
            self.recN = RNN(2*hidden, hidden, num_layers=num_layers-1, bidirectional=True, dropout=.2)
        
            
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5), 
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )
        
        # Connector: RNN第1層出力(512次元)を4096次元に拡張
        self.connector = Connector(input_dim=512, num_queries=64)
        self.llm = LLMWithLLaMA()
        
        
    def forward(self, x, y_llm=None, transcr_llm=None):
        """
        Args:
            x: 全サンプルの特徴量 (batch_size, 256, 1, width)
            y_llm: LLM用サンプルの特徴量 (llm_batch_size, 256, 1, width)
            transcr_llm: LLM用の正解文字列 (llm_batch_size,)
        """
        # RNN処理（全サンプル）
        y = x.permute(2, 3, 0, 1)[0]
        y1 = self.rec1(y)[0]

        y = self.recN(y1)[0]
        y = self.fnl(y)

        # LLM処理（選択されたサンプルのみ）
        output_llm = None
        if y_llm is not None and transcr_llm is not None and self.training:
            # y_llmからRNN第1層の出力を取得
            y_llm_seq = y_llm.permute(2, 3, 0, 1)[0]  # (width, llm_batch, 256)
            y1_llm = self.rec1(y_llm_seq)[0]  # (width, llm_batch, 512)

            # Forward/Backward方向を統合
            y1_llm = self.rnn_projection(y1_llm)  # (width, llm_batch, 512)

            # Connectorで4096次元に変換
            prefix_input = y1_llm.permute(1, 0, 2)  # (llm_batch, width, 512)
            inputs_embeds = self.connector(prefix_input)   # (llm_batch, 64, 4096)

            # テキストをトークン化（max_length=64で統一）
            llm_labels = self.llm.tokenizer(
                list(transcr_llm),
                return_tensors="pt",
                padding="max_length",  # 常に64トークンに統一
                truncation=True,
                max_length=64          # inputs_embedsと同じ長さ
            )
            labels = llm_labels["input_ids"].to(y_llm.device)  # (llm_batch, 64)

            # LLM呼び出し（シンプルに！）
            output_llm = self.llm(
                inputs_embeds=inputs_embeds.half(),  # (batch, 64, 4096) float16に変換
                labels=labels                         # (batch, 64) ← 長さ一致！
            )

        if self.training:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0], output_llm
        else:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]


class HTRNet(nn.Module):
    def __init__(self, arch_cfg, nclasses):
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
            self.top = CTCtopB(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)

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
        y = self.features(x)

        # LLM用サンプルの特徴量抽出
        y_llm = None
        if img_llm is not None:
            if self.stn is not None:
                img_llm = self.stn(img_llm)
            y_llm = self.features(img_llm)

        # CTCtopBに渡す
        if transcr_llm is not None:
            y = self.top(y, y_llm=y_llm, transcr_llm=transcr_llm)
        else:
            y = self.top(y)

        return y