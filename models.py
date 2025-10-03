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


class Connector1D(nn.Module):
    """(B, T, D_in) -> (B, T, d_llm)"""
    def __init__(self, d_in, d_llm=512):
        super().__init__()
        self.proj = nn.Linear(d_in, d_llm)
        self.ln = nn.LayerNorm(d_llm)
    def forward(self, x):  # x: (B, T, D_in)
        return self.ln(self.proj(x))
    
    
class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru',d_llm=512, enable_connector=True):
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
        
        
        self.connector = Connector1D(2*hidden, d_llm) if enable_connector else None
        self.llm_prefix = None  # 学習ループから参照するための置き場
        
        
    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y1 = self.rec1(y)[0]
        
        if (self.connector is not None) and self.training:
            y1_bt = y1.permute(1, 0, 2).contiguous()   # (B, T, 2H)
            self.llm_prefix = self.connector(y1_bt)    # (B, T, d_llm)
        else:
            self.llm_prefix = None
            
        y = self.recN(y1)[0]
        y = self.fnl(y)

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

        self.features = HybridBackboneCRNNMobileViT(flattening=arch_cfg.flattening)

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
    def __init__(self, flattening='maxpool'):
        super().__init__()
        self.flattening = flattening

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(BasicBlock(32,64), BasicBlock(64,64))
        self.pool1 = nn.MaxPool2d(2,2)
        self.mvit1 = MobileViTBlock(64, d_model=80, heads=8, num_layers=1, mlp_dim=160, patch=4)

        self.res2 = nn.Sequential(*[BasicBlock(64,128)]+[BasicBlock(128,128)]*3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.mvit2 = MobileViTBlock(128, d_model=80, heads=8, num_layers=1, mlp_dim=160, patch=8)

        self.res3 = nn.Sequential(*[BasicBlock(128,256)]+[BasicBlock(256,256)]*3)
        self.out_channels = 256

    def forward(self, x):
        y = self.stem(x)          # (B,32,64,512)
        y = self.res1(y)          # (B,64,64,512)
        y = self.pool1(y)         # (B,64,32,256)
        y = self.mvit1(y)         # (B,64,32,256)
        y = self.res2(y)          # (B,128,32,256)
        y = self.pool2(y)         # (B,128,16,128)
        y = self.mvit2(y)         # (B,128,16,128)
        y = self.res3(y)          # (B,256,16,128)

        # Column MaxPool → (B,256,1,128)
        if self.flattening == 'maxpool':
            y = F.max_pool2d(y, kernel_size=(y.size(2), 1), stride=(y.size(2), 1))
        else:
            y = F.avg_pool2d(y, kernel_size=(y.size(2), 1), stride=(y.size(2), 1))
        return y
