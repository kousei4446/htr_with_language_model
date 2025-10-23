import torch.nn as nn
import torch.nn.functional as F
import torch

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



class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [4, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1

        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
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



class Connector1D(nn.Module):
    """
    (B, T, D_in) -> (B, T//ds_factor, d_llm)
    - 分岐なし
    - ds_factor=1 なら T はそのまま（Conv1d stride=1）
    """
    def __init__(self, d_in, d_llm=512, ds_factor=2, mid_mult=2, use_residual=True):
        super().__init__()
        self.ds_factor = int(ds_factor)
        d_mid = max(d_in, d_in * mid_mult)

        # 学習的ダウンサンプル（ds=1 なら長さ不変）
        self.down = nn.Sequential(
            nn.Conv1d(d_in, d_mid, kernel_size=5, stride=self.ds_factor, padding=2, bias=False),
            nn.BatchNorm1d(d_mid),
            nn.GELU(),
            nn.Conv1d(d_mid, d_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_in),
        )

        # 残差経路：AvgPool の kernel=stride=ds なので ds=1 なら恒等
        self.skip = nn.AvgPool1d(kernel_size=self.ds_factor, stride=self.ds_factor) if use_residual else nn.Identity()

        # LLM 向け投影
        self.proj = nn.Linear(d_in, d_llm)
        self.ln = nn.LayerNorm(d_llm)

    def forward(self, x):                   # x: (B, T, D_in)
        h = self.down(x.transpose(1, 2))    # (B, D, T//ds)
        s = self.skip(x.transpose(1, 2))    # (B, D, T//ds)
        y = F.gelu(h + s).transpose(1, 2)   # (B, T//ds, D_in)
        return self.ln(self.proj(y))        # (B, T//ds, d_llm)
    
    
class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru',d_llm=2048, enable_connector=True):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        RNN = nn.GRU if rnn_type == 'gru' else nn.LSTM

        self.rec1 = RNN(input_size, hidden, num_layers=1, bidirectional=True, dropout=0.0)

        self.recN = None
        if num_layers > 1:
            self.recN = RNN(2*hidden, hidden, num_layers=num_layers-1, bidirectional=True, dropout=.2)

        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5),
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

        # ★ Connector×2: CNN末とRNN中段
        if enable_connector:
            self.connector1 = Connector1D(input_size, d_llm, ds_factor=1)  # CNN末（長さ維持）
            self.connector2 = Connector1D(2*hidden, d_llm, ds_factor=1)    # RNN中段
        else:
            self.connector1 = None
            self.connector2 = None

        self.llm_prefix1 = None  # CNN末からのprefix
        self.llm_prefix2 = None  # RNN中段からのprefix
        
        
    def forward(self, x):
        # x: (B, C, H, W) - CNN出力

        # ★ Connector1: CNN末（RNN前）
        if (self.connector1 is not None) and self.training:
            # x: (B, C, H, W) → (B, W, C) でH次元を潰す
            x_flat = x.squeeze(2) if x.size(2) == 1 else torch.max(x, dim=2)[0]  # (B, C, W)
            x_bt = x_flat.permute(0, 2, 1).contiguous()  # (B, W, C)
            self.llm_prefix1 = self.connector1(x_bt)  # (B, W, d_llm)
        else:
            self.llm_prefix1 = None

        y = x.permute(2, 3, 0, 1)[0]  # (W, B, C)
        y1 = self.rec1(y)[0]          # (W, B, 2H)

        # ★ Connector2: RNN中段（rec1後）
        if (self.connector2 is not None) and self.training:
            y1_bt = y1.permute(1, 0, 2).contiguous()  # (B, W, 2H)
            self.llm_prefix2 = self.connector2(y1_bt)  # (B, W, d_llm)
        else:
            self.llm_prefix2 = None

        y = self.recN(y1)[0]  # (W, B, 2H)
        y = self.fnl(y)       # (W, B, nclasses)

        if self.training:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]
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

        self.features = CNN(cnn_cfg, flattening=arch_cfg.flattening)

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

    def forward(self, x):

        if self.stn is not None:
            x = self.stn(x)

        y = self.features(x)
        y = self.top(y)

        return y
    
    