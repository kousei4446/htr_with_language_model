"""
TensorBoardロギング用モジュール
LLM学習の監視に特化した機能を提供
"""
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch


class HTRLogger:
    """HTR学習用のTensorBoardロガー"""

    def __init__(self, log_dir=None, config=None):
        """
        Args:
            log_dir: ログ保存先（Noneなら自動生成）
            config: 設定オブジェクト（ハイパーパラメータ記録用）
        """
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join('runs', f'experiment_{timestamp}')

        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        print('=' * 60)
        print(f'TensorBoard logging to: {log_dir}')
        print(f'Start TensorBoard with: tensorboard --logdir=runs')
        print('=' * 60)

        # グローバルステップカウンタ
        self.global_step = 0

        # Epoch内の統計用バッファ
        self.reset_epoch_stats()

        # 設定を記録
        if config is not None:
            self.log_hparams(config)

    def reset_epoch_stats(self):
        """Epoch開始時に統計をリセット"""
        self.epoch_losses = {
            'total': [],
            'ctc': [],
            'aux': [],
            'bilstm_layer1': [],  # BiLSTM layer1 CTC lossが計算された場合のみ追加
            'llm': [],  # LLM lossが計算された場合のみ追加
            'roberta': [],  # RoBERTa補助lossが計算された場合のみ追加
            'pll_bilstm': [],  # BiLSTM layer1 PLL lossが計算された場合のみ追加
            'pll_mobilevit': [],  # MobileViT PLL lossが計算された場合のみ追加
            'lm': []  # Language Model lossが計算された場合のみ追加
        }

    def log_hparams(self, config):
        """ハイパーパラメータを記録"""
        hparams = {
            'lr': config.train.lr,
            'batch_size': config.train.batch_size,
            'num_epochs': config.train.num_epochs,
            'llm_sample_ratio': config.train.get('llm_sample_ratio', 0.125),
            'rnn_hidden_size': config.arch.rnn_hidden_size,
            'rnn_layers': config.arch.rnn_layers,
        }
        # ハイパーパラメータをテキストとして記録
        hparam_str = '\n'.join([f'{k}: {v}' for k, v in hparams.items()])
        self.writer.add_text('config/hyperparameters', hparam_str, 0)

    def log_batch_loss(self, epoch, total_loss, ctc_loss=None, aux_loss=None, llm_loss=None):
        """
        バッチごとのlossを記録

        Args:
            epoch: 現在のepoch番号
            total_loss: 総損失（必須）
            ctc_loss: CTC損失（オプション）
            aux_loss: 補助損失（オプション）
            llm_loss: LLM損失（オプション、1/8のサンプルでのみ計算される）
        """
        # グローバルステップをインクリメント
        self.global_step += 1

        # 総損失（常に記録）
        self.writer.add_scalar('train/total_loss', total_loss, self.global_step)
        self.epoch_losses['total'].append(total_loss)

        # CTC損失
        if ctc_loss is not None:
            self.writer.add_scalar('train/ctc_loss', ctc_loss, self.global_step)
            self.epoch_losses['ctc'].append(ctc_loss)

        # 補助損失（CNN shortcut）
        if aux_loss is not None:
            self.writer.add_scalar('train/aux_loss', aux_loss, self.global_step)
            self.epoch_losses['aux'].append(aux_loss)

        # LLM損失（重要：1/8のサンプルでのみ計算される）
        if llm_loss is not None and llm_loss > 0:
            self.writer.add_scalar('train/llm_loss', llm_loss, self.global_step)
            self.epoch_losses['llm'].append(llm_loss)
            # LLM lossが計算された頻度も記録
            self.writer.add_scalar('train/llm_computed', 1.0, self.global_step)
        else:
            # LLM lossが計算されなかったことを記録（0を追加）
            self.writer.add_scalar('train/llm_computed', 0.0, self.global_step)

    def log_epoch_summary(self, epoch):
        """
        Epoch終了時の平均lossを記録

        Args:
            epoch: 現在のepoch番号
        """
        # 各損失の平均を計算
        if len(self.epoch_losses['total']) > 0:
            avg_total = sum(self.epoch_losses['total']) / len(self.epoch_losses['total'])
            self.writer.add_scalar('train_epoch/avg_total_loss', avg_total, epoch)

        if len(self.epoch_losses['ctc']) > 0:
            avg_ctc = sum(self.epoch_losses['ctc']) / len(self.epoch_losses['ctc'])
            self.writer.add_scalar('train_epoch/avg_ctc_loss', avg_ctc, epoch)

        if len(self.epoch_losses['aux']) > 0:
            avg_aux = sum(self.epoch_losses['aux']) / len(self.epoch_losses['aux'])
            self.writer.add_scalar('train_epoch/avg_aux_loss', avg_aux, epoch)

        # LLM損失の平均（計算された場合のみ）
        if len(self.epoch_losses['llm']) > 0:
            avg_llm = sum(self.epoch_losses['llm']) / len(self.epoch_losses['llm'])
            self.writer.add_scalar('train_epoch/avg_llm_loss', avg_llm, epoch)
            # LLM lossが計算された割合
            llm_ratio = len(self.epoch_losses['llm']) / len(self.epoch_losses['total'])
            self.writer.add_scalar('train_epoch/llm_computation_ratio', llm_ratio, epoch)

            print(f'[Epoch {epoch}] LLM loss computed in {llm_ratio:.1%} of batches')

        # RoBERTa補助損失の平均（計算された場合のみ）
        if len(self.epoch_losses['roberta']) > 0:
            avg_roberta = sum(self.epoch_losses['roberta']) / len(self.epoch_losses['roberta'])
            self.writer.add_scalar('train_epoch/avg_roberta_loss', avg_roberta, epoch)
            # RoBERTa lossが計算された割合
            roberta_ratio = len(self.epoch_losses['roberta']) / len(self.epoch_losses['total'])
            self.writer.add_scalar('train_epoch/roberta_computation_ratio', roberta_ratio, epoch)

            print(f'[Epoch {epoch}] RoBERTa loss computed in {roberta_ratio:.1%} of batches, avg={avg_roberta:.4f}')

        # BiLSTM layer1 PLL損失の平均（計算された場合のみ）
        if len(self.epoch_losses['pll_bilstm']) > 0:
            avg_pll_bilstm = sum(self.epoch_losses['pll_bilstm']) / len(self.epoch_losses['pll_bilstm'])
            self.writer.add_scalar('train_epoch/avg_pll_bilstm_loss', avg_pll_bilstm, epoch)
            # BiLSTM PLL lossが計算された割合
            pll_bilstm_ratio = len(self.epoch_losses['pll_bilstm']) / len(self.epoch_losses['total'])
            self.writer.add_scalar('train_epoch/pll_bilstm_computation_ratio', pll_bilstm_ratio, epoch)

            print(f'[Epoch {epoch}] BiLSTM PLL loss computed in {pll_bilstm_ratio:.1%} of batches, avg={avg_pll_bilstm:.4f}')

        # MobileViT PLL損失の平均（計算された場合のみ）
        if len(self.epoch_losses['pll_mobilevit']) > 0:
            avg_pll_mobilevit = sum(self.epoch_losses['pll_mobilevit']) / len(self.epoch_losses['pll_mobilevit'])
            self.writer.add_scalar('train_epoch/avg_pll_mobilevit_loss', avg_pll_mobilevit, epoch)
            # MobileViT PLL lossが計算された割合
            pll_mobilevit_ratio = len(self.epoch_losses['pll_mobilevit']) / len(self.epoch_losses['total'])
            self.writer.add_scalar('train_epoch/pll_mobilevit_computation_ratio', pll_mobilevit_ratio, epoch)

            print(f'[Epoch {epoch}] MobileViT PLL loss computed in {pll_mobilevit_ratio:.1%} of batches, avg={avg_pll_mobilevit:.4f}')

        # Language Model損失の平均（計算された場合のみ）
        if len(self.epoch_losses['lm']) > 0:
            avg_lm = sum(self.epoch_losses['lm']) / len(self.epoch_losses['lm'])
            self.writer.add_scalar('train_epoch/avg_lm_loss', avg_lm, epoch)
            # LM lossが計算された割合
            lm_ratio = len(self.epoch_losses['lm']) / len(self.epoch_losses['total'])
            self.writer.add_scalar('train_epoch/lm_computation_ratio', lm_ratio, epoch)

            print(f'[Epoch {epoch}] LM loss computed in {lm_ratio:.1%} of batches, avg={avg_lm:.4f}')

        # 次のepochのためにリセット
        self.reset_epoch_stats()

    def log_learning_rate(self, epoch, lr):
        """
        学習率を記録

        Args:
            epoch: 現在のepoch番号
            lr: 現在の学習率
        """
        self.writer.add_scalar('train/learning_rate', lr, epoch)

    def log_metrics(self, epoch, cer, wer, split='val'):
        """
        評価指標（CER/WER）を記録

        Args:
            epoch: 現在のepoch番号
            cer: Character Error Rate
            wer: Word Error Rate
            split: 'val' または 'test'
        """
        self.writer.add_scalar(f'{split}/cer', cer, epoch)
        self.writer.add_scalar(f'{split}/wer', wer, epoch)

    def log_sample_text(self, epoch, ground_truth, prediction, split='val'):
        """
        サンプルテキスト（予測結果）を記録

        Args:
            epoch: 現在のepoch番号
            ground_truth: 正解テキスト
            prediction: 予測テキスト
            split: 'val' または 'test'
        """
        text = f"Ground Truth: {ground_truth}\nPrediction:   {prediction}"
        self.writer.add_text(f'{split}/sample_prediction', text, epoch)

    def log_model_graph(self, model, input_shape):
        """
        モデルグラフを記録

        Args:
            model: PyTorchモデル
            input_shape: 入力の形状（例: (1, 1, 128, 1024)）
        """
        try:
            dummy_input = torch.randn(input_shape)
            self.writer.add_graph(model, dummy_input)
            print('Model graph logged to TensorBoard')
        except Exception as e:
            print(f'Warning: Could not log model graph: {e}')

    def close(self):
        """ロガーを閉じる"""
        self.writer.close()
        print(f'TensorBoard logging closed. Logs saved to: {self.log_dir}')


class LLMLossTracker:
    """LLM損失の詳細な追跡用ヘルパークラス"""

    def __init__(self, llm_sample_ratio=0.125):
        """
        Args:
            llm_sample_ratio: LLM処理されるサンプルの割合（デフォルト: 1/8）
        """
        self.llm_sample_ratio = llm_sample_ratio
        self.llm_losses = []
        self.total_batches = 0
        self.llm_computed_batches = 0

    def update(self, llm_loss):
        """
        LLM損失を記録

        Args:
            llm_loss: LLM損失（Noneまたは0の場合は計算されなかった）
        """
        self.total_batches += 1

        if llm_loss is not None and llm_loss > 0:
            self.llm_losses.append(llm_loss)
            self.llm_computed_batches += 1

    def get_average(self):
        """平均LLM損失を取得"""
        if len(self.llm_losses) == 0:
            return 0.0
        return sum(self.llm_losses) / len(self.llm_losses)

    def get_computation_ratio(self):
        """LLM損失が計算された割合を取得"""
        if self.total_batches == 0:
            return 0.0
        return self.llm_computed_batches / self.total_batches

    def get_stats(self):
        """統計情報を取得"""
        return {
            'avg_loss': self.get_average(),
            'computation_ratio': self.get_computation_ratio(),
            'total_batches': self.total_batches,
            'llm_computed_batches': self.llm_computed_batches,
            'expected_ratio': self.llm_sample_ratio
        }

    def reset(self):
        """統計をリセット"""
        self.llm_losses = []
        self.total_batches = 0
        self.llm_computed_batches = 0
