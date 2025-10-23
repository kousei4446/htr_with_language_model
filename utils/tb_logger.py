# utils/tb_logger.py
from torch.utils.tensorboard import SummaryWriter
import os
import datetime as dt

class TBLogger:
    def __init__(self, log_dir="runs/htr", enabled=True, rank=0):
        self.enabled = enabled and (rank == 0)  # DDP想定時のrankガード
        self.writer = SummaryWriter(log_dir=self._make_logdir(log_dir)) if self.enabled else None
        self.global_step = 0

    def _make_logdir(self, base):
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(base, ts)

    def log_train_step(self, ctc_only, llm_loss, total, stage, ratio_ema=None, grad_ema=None, ce_loss=None, kl_loss=None):
        if not self.enabled: return
        w = self.writer
        w.add_scalar('loss/ctc_only', ctc_only, self.global_step)
        w.add_scalar('loss/llm', llm_loss, self.global_step)
        w.add_scalar('loss/total', total, self.global_step)
        w.add_scalar('monitor/stage', stage, self.global_step)
        if ratio_ema is not None: w.add_scalar('monitor/ratio_ema', ratio_ema, self.global_step)
        if grad_ema  is not None: w.add_scalar('monitor/grad_ema',  grad_ema,  self.global_step)
        # ★ LAIL損失の内訳
        if ce_loss is not None: w.add_scalar('loss/llm_ce', ce_loss, self.global_step)
        if kl_loss is not None: w.add_scalar('loss/llm_kl', kl_loss, self.global_step)
        self.global_step += 1

    def log_eval(self, epoch, cer=None, wer=None, split="val"):
        if not self.enabled: return
        if cer is not None: self.writer.add_scalar(f'eval/{split}_cer', cer, epoch)
        if wer is not None: self.writer.add_scalar(f'eval/{split}_wer', wer, epoch)

    def log_dep_ratio(self, epoch, dep_ratio):
        if not self.enabled: return
        self.writer.add_scalar('monitor/dep_ratio', dep_ratio, epoch)

    def close(self):
        if self.enabled:
            self.writer.close()
