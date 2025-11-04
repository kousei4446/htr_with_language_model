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

    def forward(self, x, reduce='max', return_mobilevit=False):

        y = x
        mobilevit1_output = None  # MobileViT1層の出力を保存
        mobilevit2_output = None  # MobileViT2層の出力を保存

        for i, nn_module in enumerate(self.features):
            y = nn_module(y)
            # 各MobileViT層の出力を保存
            if isinstance(nn_module, MobileViTBlock):
                # モジュール名を取得してどのMobileViT層か判定
                module_name = list(self.features._modules.keys())[i]
                if 'mvit1' in module_name:
                    mobilevit1_output = y
                elif 'mvit2' in module_name:
                    mobilevit2_output = y

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        if return_mobilevit:
            return y, mobilevit1_output, mobilevit2_output
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
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.projection(x)   # (batch, 128, 3072) - 次元拡張
        return x


class ConnectorMobileViT1(nn.Module):
    """MobileViT1出力用のConnector (64次元 → 3072次元)

    入力: (batch, 64, H, W) - mobilevit1層の出力
    出力: (batch, seq_len, 3072) - LLM用の埋め込み

    パラメータ数: 約1.9M
    - 64 → 512: 33K params
    - 512 → 3072: 1.6M params
    """
    def __init__(self, input_dim=64, output_dim=3072):
        super().__init__()
        # Projection: 64 → 512 → 3072
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (batch, 64, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (batch, 64, H*W)
        x = x.permute(0, 2, 1)  # (batch, H*W, 64)
        x = self.projection(x)  # (batch, H*W, 3072)
        return x


class ConnectorMobileViT2(nn.Module):
    """MobileViT2出力用のConnector (128次元 → 3072次元)

    入力: (batch, 128, H, W) - mobilevit2層の出力
    出力: (batch, seq_len, 3072) - LLM用の埋め込み

    パラメータ数: 約2.2M
    - 128 → 512: 66K params
    - 512 → 3072: 1.6M params
    """
    def __init__(self, input_dim=128, output_dim=3072):
        super().__init__()
        # Projection: 128 → 512 → 3072
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (batch, 128, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (batch, 128, H*W)
        x = x.permute(0, 2, 1)  # (batch, H*W, 128)
        x = self.projection(x)  # (batch, H*W, 3072)
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

        print(f"📦 Loading model: {model_name}")

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

        print(f"✅ Model loaded successfully!")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Vocab size: {self.config.vocab_size}")
        print(f"   Initial device: CPU (will move to GPU with net.to(device))")

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
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru', d_llm=512, enable_connector=True, use_llm=False, llm_source='rnn'):
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
        self.llm_source = llm_source  # 'rnn', 'mobilevit1', 'mobilevit2', 'all'

        if use_llm:
            # ✅ LLMは1つだけ作成（共有）
            print("🔥 Loading shared LLM (LLaMA-3.2-3B)...")
            self.llm = LLMWithLLaMA()

            # ✅ Connectorは個別に作成（学習可能パラメータ）
            if 'rnn' in llm_source or llm_source == 'all':
                print("   - Loading Connector_RNN (512→3072)...")
                self.connector_rnn = Connector(input_dim=512)
            else:
                self.connector_rnn = None

            if 'mobilevit1' in llm_source or llm_source == 'all':
                print("   - Loading Connector_MV1 (64→3072)...")
                self.connector_mv1 = ConnectorMobileViT1(input_dim=64)
            else:
                self.connector_mv1 = None

            if 'mobilevit2' in llm_source or llm_source == 'all':
                print("   - Loading Connector_MV2 (128→3072)...")
                self.connector_mv2 = ConnectorMobileViT2(input_dim=128)
            else:
                self.connector_mv2 = None
        else:
            print("⚡ LLM disabled: Using CNN shortcut only")
            self.llm = None
            self.connector_rnn = None
            self.connector_mv1 = None
            self.connector_mv2 = None
        
        
    def forward(self, x, y_llm=None, mv1_llm=None, mv2_llm=None, transcr_llm=None):
        """
        Args:
            x: 全サンプルの特徴量 (batch_size, 256, 1, width)
            y_llm: RNN用LLMサンプルの特徴量 (llm_batch_size, 256, 1, width)
            mv1_llm: MobileViT1用LLMサンプルの出力 (llm_batch_size, 64, H1, W1)
            mv2_llm: MobileViT2用LLMサンプルの出力 (llm_batch_size, 128, H2, W2)
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

        # ✅ LLM処理（逐次実行で1つのLLMを共有）
        output_llm_rnn = None
        output_llm_mv1 = None
        output_llm_mv2 = None

        if self.use_llm and transcr_llm is not None and self.training:
            # RNN経路（順番1）
            if self.connector_rnn is not None and y_llm is not None:
                # y_llmからRNN layer1の出力を取得
                y_llm_seq = y_llm.permute(2, 3, 0, 1)[0]  # (width, llm_batch, 256)
                y1_llm = self.rec1(y_llm_seq)[0]  # (width, llm_batch, 512) - layer1 output only

                # Connectorで3072次元に変換
                prefix_input = y1_llm.permute(1, 0, 2)  # (llm_batch, width, 512)
                inputs_embeds = self.connector_rnn(prefix_input)   # (llm_batch, width, 3072)

                # テキストをトークン化
                llm_labels = self.llm.tokenizer(  # ✅ 共有LLM
                    list(transcr_llm),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=inputs_embeds.shape[1]
                )
                labels = llm_labels["input_ids"].to(y_llm.device)

                output_llm_rnn = self.llm(  # ✅ 共有LLM
                    inputs_embeds=inputs_embeds.half(),
                    labels=labels
                )

            # MobileViT1経路（順番2）
            if self.connector_mv1 is not None and mv1_llm is not None:
                # MobileViT1出力をConnectorで変換
                inputs_embeds = self.connector_mv1(mv1_llm)  # (llm_batch, H1*W1, 3072)

                # テキストをトークン化
                llm_labels = self.llm.tokenizer(  
                    list(transcr_llm),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=inputs_embeds.shape[1]
                )
                labels = llm_labels["input_ids"].to(mv1_llm.device)

                output_llm_mv1 = self.llm(  
                    inputs_embeds=inputs_embeds.half(),
                    labels=labels
                )

            # MobileViT2経路（順番3）
            if self.connector_mv2 is not None and mv2_llm is not None:
                # MobileViT2出力をConnectorで変換
                inputs_embeds = self.connector_mv2(mv2_llm)  # (llm_batch, H2*W2, 3072)

                # テキストをトークン化
                llm_labels = self.llm.tokenizer(  
                    list(transcr_llm),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=inputs_embeds.shape[1]
                )
                labels = llm_labels["input_ids"].to(mv2_llm.device)

                output_llm_mv2 = self.llm(  
                    inputs_embeds=inputs_embeds.half(),
                    labels=labels
                )

        if self.training:
            return y_ctc, self.cnn(x).permute(2, 3, 0, 1)[0], output_llm_rnn, output_llm_mv1, output_llm_mv2
        else:
            return y_ctc, self.cnn(x).permute(2, 3, 0, 1)[0]


class HTRNet(nn.Module):
    def __init__(self, arch_cfg, nclasses, use_llm=False, llm_source='rnn'):
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

        self.llm_source = llm_source  # 'rnn', 'mobilevit1', 'mobilevit2', 'all'

        head = arch_cfg.head_type
        if head=='cnn':
            self.top = CTCtopC(hidden, nclasses)
        elif head=='rnn':
            self.top = CTCtopR(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)
        elif head=='both':
            self.top = CTCtopB(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type, use_llm=use_llm, llm_source=llm_source)

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
        mv1_llm = None
        mv2_llm = None

        if img_llm is not None:
            if self.stn is not None:
                img_llm = self.stn(img_llm)

            # MobileViT出力が必要か判定
            need_mv = ('mobilevit1' in self.llm_source or
                      'mobilevit2' in self.llm_source or
                      self.llm_source == 'all')

            if need_mv:
                y_llm, mv1_llm, mv2_llm = self.features(img_llm, return_mobilevit=True)
            else:
                y_llm = self.features(img_llm)

        # CTCtopBに渡す
        if transcr_llm is not None:
            y = self.top(y, y_llm=y_llm, mv1_llm=mv1_llm, mv2_llm=mv2_llm, transcr_llm=transcr_llm)
        else:
            y = self.top(y)

        return y