"""
デバッグ用スクリプト: Connector の形状確認
LLMは一切読み込まず、QFormer/Connectorの入出力形状のみテスト
"""

import torch
import torch.nn as nn

# models.py から必要なクラスのみコピー
class QFormer(nn.Module):
    """BLIP-2スタイルのQ-Former: 128トークン→20トークンに圧縮

    FFNで直接LLM次元(3072)に変換することで、後段のexpansion層を削減
    """
    def __init__(self, input_dim=512, num_queries=20, num_heads=8, output_dim=3072):
        super().__init__()

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, input_dim))

        # Self-attention (クエリ同士が情報を共有)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network: 512 → 2048 → 3072 (直接LLM次元に変換)
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),  # 512 → 2048
            nn.GELU(),
            nn.Linear(input_dim * 4, output_dim),  # 2048 → 3072
            nn.LayerNorm(output_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch, 128, 512)
        batch_size = x.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        # (batch, 20, 512)

        # Self-attention: queries attend to each other
        self_attn_out, _ = self.self_attn(queries, queries, queries)
        queries = queries + self_attn_out  # Residual connection
        queries = self.norm1(queries)

        # Cross-attention: queries attend to x
        cross_attn_out, _ = self.cross_attn(queries, x, x)
        queries = queries + cross_attn_out  # Residual connection
        queries = self.norm2(queries)

        # Feed-forward: 512 → 3072 (直接LLM次元に変換)
        output = self.ffn(queries)  # Residual接続なし（次元が変わるため）

        return output  # (batch, 20, 3072)


class Connector(nn.Module):
    """Q-Formerベースのコネクタ（Llama-3.2-3B用）

    改善点:
    - トークン数: 128 → 20 (84%削減)
    - Self-attention追加で情報損失を軽減
    - FFNで直接3072次元に変換（expansion層削減）
    - パラメータ数: 約7.5M (旧版11.16Mから33%削減)
    """
    def __init__(self, input_dim=512, num_queries=20, output_dim=3072):
        super().__init__()

        # Q-Former: 128トークン → 20トークンに圧縮 + 3072次元に変換
        self.qformer = QFormer(
            input_dim=input_dim,
            num_queries=num_queries,
            num_heads=8,
            output_dim=output_dim
        )

    def forward(self, x):
        # x: (batch, 128, 512)
        x = self.qformer(x)      # (batch, 20, 3072)
        return x


def test_connector_shapes():
    """Connector の形状テスト"""
    print("=" * 60)
    print("Connector 形状テスト開始")
    print("=" * 60)

    # Connector インスタンス作成
    connector = Connector(input_dim=512, num_queries=20, output_dim=3072)
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
    print(f"期待形状: ({batch_size}, 20, 3072)")
    print(f"✅ 正常" if output1.shape == torch.Size([batch_size, 20, 3072]) else "❌ エラー")

    # テストケース2: width=256 (エラーメッセージから推測される実際の入力)
    print("\n[テスト2] width=256 (実際に来ている可能性がある入力)")
    width = 256

    x2 = torch.randn(batch_size, width, input_dim)
    print(f"入力形状: {x2.shape} (batch={batch_size}, width={width}, dim={input_dim})")

    with torch.no_grad():
        output2 = connector(x2)

    print(f"出力形状: {output2.shape}")
    print(f"期待形状: ({batch_size}, 20, 3072)")
    print(f"✅ 正常" if output2.shape == torch.Size([batch_size, 20, 3072]) else "❌ エラー")

    # パラメータ数確認
    print("\n[パラメータ数]")
    total_params = sum(p.numel() for p in connector.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    test_connector_shapes()
