
class StageMonitor:
    def __init__(self, ema_decay=0.97, hold_epochs=1,
                 ratio_band=(0.2, 0.8), grad_band=(1e-5, 1e-2),
                 dep_ratio_stage12=1.5, dep_ratio_stage23=3.0,
                 passes_needed=2):
        self.decay = ema_decay
        self.hold_epochs = hold_epochs
        self.ratio_lo, self.ratio_hi = ratio_band
        self.grad_lo, self.grad_hi = grad_band
        self.dep12 = dep_ratio_stage12
        self.dep23 = dep_ratio_stage23
        self.passes_needed = passes_needed
        self.reset()
        
    def reset(self):
        self.ctc_ema = self.llm_ema = self.ratio_ema = self.grad_ema = None
        self.stage, self.passes, self.last_stage_change_epoch = 1, 0, -10
        self.last_eval = {}
        
    def _ema(self, old, new):
        return float(new) if old is None else float(self.decay*old + (1-self.decay)*new)
    
    def update_batch(self, ctc_loss, llm_loss, connector_grad_mean, alpha_llm=0.5, k=1):
        self.ctc_ema = self._ema(self.ctc_ema, float(ctc_loss))
        self.llm_ema = self._ema(self.llm_ema, float(llm_loss))
        if self.ctc_ema and self.ctc_ema > 0:
            ratio = (alpha_llm * k * float(llm_loss)) / float(ctc_loss)
            self.ratio_ema = self._ema(self.ratio_ema, ratio)
        if connector_grad_mean is not None:
            self.grad_ema = self._ema(self.grad_ema, float(connector_grad_mean))
    
    def on_eval_dependency(self, dep_ratio, cer=None, wer=None):
        self.last_eval = {'dep_ratio': dep_ratio, 'cer': cer, 'wer': wer,
                          'ctc_ema': self.ctc_ema, 'llm_ema': self.llm_ema,
                          'ratio_ema': self.ratio_ema, 'grad_ema': self.grad_ema}
    
    def maybe_advance(self, epoch):
        if self.ratio_ema is None or self.grad_ema is None:
            return self.stage, False
        grad_ok = (self.grad_lo <= self.grad_ema <= self.grad_hi)
        ratio_ok = (self.ratio_lo <= self.ratio_ema <= self.ratio_hi)
        dep = self.last_eval.get('dep_ratio')
        if dep is None:
            return self.stage, False
        dep_ok = (dep >= self.dep12) if self.stage == 1 else (dep >= self.dep23 if self.stage == 2 else False)
        all_ok = grad_ok and ratio_ok and dep_ok and (epoch - self.last_stage_change_epoch >= self.hold_epochs)
        self.passes = self.passes + 1 if all_ok else 0
        if self.passes >= self.passes_needed and self.stage < 3:
            self.stage += 1
            self.last_stage_change_epoch = epoch
            self.passes = 0
            return self.stage, True
        return self.stage, False

