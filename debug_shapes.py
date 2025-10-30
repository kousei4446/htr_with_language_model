"""
デバッグ用スクリプト: Connector の形状確認
LLMは一切読み込まず、QFormer/Connectorの入出力形状のみテスト
"""

import torch
import torch.nn as nn

# models.py から必要なクラスのみコピー
class Connector(nn.Module):
    """Conv1dベースの学習可能なコネクタ（Llama-3.2-3B用）

    改善点:
    - Q-Former (9.5M params) → Conv1d (3.4M params) (64%削減)
    - トークン数: 128 → 21 (学習可能な圧縮)
    - 次元: 512 → 3072 (Linear projection)
    - 重要な情報を学習で保持
    """
    def __init__(self, input_dim=512, num_queries=21, output_dim=3072):
        super().__init__()

        # 学習可能な圧縮: 128 → 21
        # stride=6: 128 / 6 ≈ 21 (正確に21になる)
        self.compress = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=7, stride=6, padding=3),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )

        # Projection: 512次元 → 3072次元に拡張
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x: (batch, 128, 512)
        x = x.transpose(1, 2)    # (batch, 512, 128) - Conv1d用
        x = self.compress(x)      # (batch, 512, 21) - 学習可能な圧縮
        x = x.transpose(1, 2)    # (batch, 21, 512) - 元の形式に戻す
        x = self.projection(x)   # (batch, 21, 3072) - 次元拡張
        return x


def test_connector_shapes():
    """Connector の形状テスト"""
    print("=" * 60)
    print("Connector 形状テスト開始")
    print("=" * 60)

    # Connector インスタンス作成
    connector = Connector(input_dim=512, num_queries=21, output_dim=3072)
    connector.eval()

    # テストケース1: width=128 (期待される入力)
    print("\n[テスト1] width=128 (期待される入力)")
    batch_size = 4
    width = 128
    input_dim = 512

    x1 = torch.randn(batch_size, width, input_dim)
    print(f"入力形状: {x1.shape} (batch={batch_size}, width={width}, dim={input_dim})")

    with torch.no_grad():
        output1 = connector(x1)

    print(f"出力形状: {output1.shape}")
    print(f"期待形状: ({batch_size}, 21, 3072)")
    print(f"✅ 正常" if output1.shape == torch.Size([batch_size, 21, 3072]) else "❌ エラー")

    # テストケース2: width=256 (可変長入力のテスト)
    print("\n[テスト2] width=256 (可変長入力のテスト)")
    width = 256

    x2 = torch.randn(batch_size, width, input_dim)
    print(f"入力形状: {x2.shape} (batch={batch_size}, width={width}, dim={input_dim})")

    with torch.no_grad():
        output2 = connector(x2)

    print(f"出力形状: {output2.shape}")
    print(f"期待形状: ({batch_size}, 43, 3072) (256/6≈43)")
    print(f"✅ 正常" if output2.shape[1] == 43 and output2.shape[2] == 3072 else "❌ エラー")

    # パラメータ数確認
    print("\n[パラメータ数]")
    total_params = sum(p.numel() for p in connector.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    test_connector_shapes()
