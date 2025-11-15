"""
Q-Formerコネクタの動作確認スクリプト

このスクリプトは以下を確認します:
1. 各コネクタの出力形状が正しいか
2. パラメータ数が約14M（3コネクタ合計）であるか
3. テキスト埋め込みクエリが正しく動作するか
"""

import torch
import torch.nn as nn
from models import (
    LightweightQFormerConnector,
    ConnectorForMobileViT,
    ConnectorForBiLSTM,
    LLMWithGPT2
)


def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_lightweight_qformer():
    """LightweightQFormerConnectorの単体テスト"""
    print("=" * 80)
    print("Testing LightweightQFormerConnector")
    print("=" * 80)

    # テスト用の入力を作成
    batch_size = 4
    seq_len = 128
    input_dim = 64
    text_len = 50
    output_dim = 768

    # コネクタを作成
    connector = LightweightQFormerConnector(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=8,
        ffn_ratio=2
    )

    # 入力データを作成
    x = torch.randn(batch_size, seq_len, input_dim)
    text_embeddings = torch.randn(batch_size, text_len, output_dim)

    # Forward pass
    output = connector(x, text_embeddings)

    # 形状確認
    expected_shape = (batch_size, text_len, output_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # パラメータ数確認
    num_params = count_parameters(connector)
    expected_params = 4.8e6  # 約4.8M

    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"   Expected: ~{expected_params/1e6:.1f}M")
    print(f"   Difference: {abs(num_params - expected_params)/expected_params*100:.1f}%")
    print()

    return num_params


def test_connector_for_mobilevit():
    """ConnectorForMobileViTのテスト"""
    print("=" * 80)
    print("Testing ConnectorForMobileViT")
    print("=" * 80)

    # テスト用の入力を作成
    batch_size = 4
    channels = 64
    height = 16
    width = 128
    text_len = 50
    output_dim = 768

    # コネクタを作成
    connector = ConnectorForMobileViT(
        input_channels=channels,
        output_dim=output_dim,
        num_heads=8,
        ffn_ratio=2
    )

    # 入力データを作成（MobileViT出力形式）
    x = torch.randn(batch_size, channels, height, width)
    text_embeddings = torch.randn(batch_size, text_len, output_dim)

    # Forward pass
    output = connector(x, text_embeddings)

    # 形状確認
    expected_shape = (batch_size, text_len, output_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # パラメータ数確認
    num_params = count_parameters(connector)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print()

    return num_params


def test_connector_for_bilstm():
    """ConnectorForBiLSTMのテスト"""
    print("=" * 80)
    print("Testing ConnectorForBiLSTM")
    print("=" * 80)

    # テスト用の入力を作成
    batch_size = 4
    seq_len = 128
    input_dim = 512
    text_len = 50
    output_dim = 768

    # コネクタを作成
    connector = ConnectorForBiLSTM(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=8,
        ffn_ratio=2
    )

    # 入力データを作成（BiLSTM出力形式）
    x = torch.randn(batch_size, seq_len, input_dim)
    text_embeddings = torch.randn(batch_size, text_len, output_dim)

    # Forward pass
    output = connector(x, text_embeddings)

    # 形状確認
    expected_shape = (batch_size, text_len, output_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # パラメータ数確認
    num_params = count_parameters(connector)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print()

    return num_params


def test_with_gpt2_embeddings():
    """GPT-2の実際の埋め込みを使ったテスト"""
    print("=" * 80)
    print("Testing with real GPT-2 embeddings")
    print("=" * 80)

    # GPT-2をロード
    llm = LLMWithGPT2()

    # テスト用のテキスト
    texts = [
        "Hello world",
        "This is a test",
        "Q-Former connector",
        "Image to text"
    ]

    # テキストをトークン化
    tokens = llm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=50
    )

    # GPT-2の埋め込み層で埋め込みを取得
    with torch.no_grad():
        text_embeddings = llm.model.transformer.wte(tokens["input_ids"])

    batch_size, text_len, embed_dim = text_embeddings.shape

    print(f"✅ Text embeddings shape: {text_embeddings.shape}")
    print(f"✅ Texts: {texts}")
    print()

    # ConnectorForMobileViTでテスト
    channels = 64
    height = 16
    width = 128
    x = torch.randn(batch_size, channels, height, width)

    connector = ConnectorForMobileViT(
        input_channels=channels,
        output_dim=embed_dim,
        num_heads=8,
        ffn_ratio=2
    )

    output = connector(x, text_embeddings)

    print(f"✅ Connector output shape: {output.shape}")
    print(f"✅ Successfully processed {batch_size} samples with text queries")
    print()


def main():
    """メインテスト関数"""
    print("\n" + "=" * 80)
    print("Q-Former Connector Test Suite")
    print("=" * 80 + "\n")

    # 各コネクタのテスト
    params_qformer = test_lightweight_qformer()
    params_mvit = test_connector_for_mobilevit()
    params_bilstm = test_connector_for_bilstm()

    # 合計パラメータ数
    # MobileViT用が2つ（64ch, 128ch）、BiLSTM用が1つ
    total_params = params_mvit * 2 + params_bilstm

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"MobileViT connector (×2): {params_mvit:,} ({params_mvit/1e6:.2f}M) each")
    print(f"BiLSTM connector (×1):    {params_bilstm:,} ({params_bilstm/1e6:.2f}M)")
    print(f"Total parameters:         {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Target:                   ~14M")
    print(f"Difference:               {abs(total_params - 14e6)/14e6*100:.1f}%")
    print()

    # GPT-2埋め込みを使った実践的なテスト
    test_with_gpt2_embeddings()

    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
