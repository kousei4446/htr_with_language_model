import argparse
from omegaconf import OmegaConf

import sys
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset import HTRDataset

from models import HTRNet
from utils.transforms import aug_transforms

import torch.nn.functional as F

from utils.metrics import CER, WER
from utils.stage_monitor import StageMonitor

from models import Connector1D

from utils.tb_logger import TBLogger

# LLMモデルのインポート（グローバルスコープで必須）
from model_llm import llm_model, llm_tokenizer, llm_hidden_size

class HTRTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ★ LLM情報を初期化時に一度だけ取得
        self.device_llm = next(llm_model.parameters()).device
        self.lm_dtype = next(llm_model.parameters()).dtype
        
        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()
        self.stage_monitor = StageMonitor(
            ema_decay=getattr(config.train, 'ema_decay', 0.97),
            hold_epochs=getattr(config.train, 'hold_epochs', 1),
            ratio_band=getattr(config.train, 'ratio_band', (0.2, 0.8)),
            grad_band=getattr(config.train, 'grad_band', (1e-5, 1e-2)),
            dep_ratio_stage12=getattr(config.train, 'dep12', 1.5),
            dep_ratio_stage23=getattr(config.train, 'dep23', 3.0),
            passes_needed=getattr(config.train, 'passes_needed', 2),
        )
        self.tb = TBLogger(
            log_dir=getattr(self.config.train, 'log_dir', 'runs/htr'),
            enabled=getattr(self.config.train, 'use_tensorboard', True),
            rank=0
        )



    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=aug_transforms)
        classes = train_set.character_classes
        # print('# training lines ' + str(train_set.__len__()))

        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        # print('# validation lines ' + str(val_set.__len__()))

        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        # print('# testing lines ' + str(test_set.__len__()))

        # augmentation using data sampler
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size, 
                                  shuffle=True, num_workers=config.train.num_workers)
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)
        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)

        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        # add space to classes, if not already there
        classes += ' ' 
        classes = np.unique(classes)

        # save classes in data folder
        np.save(os.path.join(dataset_folder, 'classes.npy'), classes)

        # create dictionaries for character to index and index to character 
        # 0 index is reserved for CTC blank
        cdict = {c:(i+1) for i,c in enumerate(classes)}
        icdict = {(i+1):c for i,c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }
        # print('[DEBUG] N_train =', len(train_set))
        # print('[DEBUG] train.batch_size (cfg) =', config.train.batch_size)  
        # print('[DEBUG] steps_per_epoch (len(loader)) =', len(train_loader))

    def prepare_net(self):

        config = self.config

        device = config.device

        # print('Preparing Net - Architectural elements:')
        # print(config.arch)

        classes = self.classes['classes']

        net = HTRNet(config.arch, len(classes) + 1)
        
        if config.resume is not None:
            print('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume)
            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)
        # ★ CTCtopB かつ Connector×2 有効なら LLM次元に合わせて差し替え＆デバイス移動
        if hasattr(net.top, 'connector1') and (net.top.connector1 is not None):
            # Connector1 (CNN末)
            if net.top.connector1.ln.normalized_shape[0] != llm_hidden_size:
                net.top.connector1 = Connector1D(
                    d_in=net.top.connector1.proj.in_features,
                    d_llm=llm_hidden_size,
                    ds_factor=1
                )
            net.top.connector1.to(self.device_llm)

        if hasattr(net.top, 'connector2') and (net.top.connector2 is not None):
            # Connector2 (RNN中段)
            if net.top.connector2.ln.normalized_shape[0] != llm_hidden_size:
                net.top.connector2 = Connector1D(
                    d_in=net.top.connector2.proj.in_features,
                    d_llm=llm_hidden_size,
                    ds_factor=1
                )
            net.top.connector2.to(self.device_llm)

        self.net = net

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /self.config.train.batch_size
    
    
    def ctc_loss_safe(self, logp, labels, act_lens, label_lens):
        """cuDNN無効化を削除"""
        crit = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        try:
            # ★ cuDNN無効化を削除 → 高速化！
            return crit(logp, labels, act_lens, label_lens) / self.config.train.batch_size
        except RuntimeError:
            return crit(logp.cpu(), labels, act_lens, label_lens) / self.config.train.batch_size

    def prepare_optimizers(self):
        config = self.config
        optimizer = torch.optim.AdamW(self.net.parameters(), config.train.lr, weight_decay=0.00005)

        self.optimizer = optimizer

        max_epochs = config.train.num_epochs
        if config.train.scheduler == 'mstep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
        else:
            raise NotImplementedError('Alternative schedulers not implemented yet')

    def decode(self, tdec, tdict, blank_id=0):
        import numpy as np
        tdec = np.asarray(tdec)
        if tdec.ndim == 0:              # スカラーなら長さ1に
            tdec = tdec.reshape(1)

        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[int(t)] for t in tt if int(t) != blank_id])
        return dec_transcr

                
    def sample_decoding(self):

        # get a random image from the test set
        img, transcr = self.loaders['val'].dataset[np.random.randint(0, len(self.loaders['val'].dataset))]

        img = img.unsqueeze(0).to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)
            if self.config.arch.head_type == 'both':
                tst_o = tst_o[0]

        self.net.train()

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # remove duplicates
        dec_transcr = self.decode(tdec, self.classes['i2c'])

        print('orig:: ' + transcr.strip())
        print('pred:: ' + dec_transcr.strip())

    def build_stage_settings(self):
        """
        ステージ切替えの閾値は必要最小限：epoch で分岐にします。
        - 例: 0〜2: Stage1, 3〜5: Stage2, 6〜: Stage3
        必要なら config.train.stage2_start_epoch / stage3_start_epoch を作って使ってください。
        """
        s = self.stage_monitor.stage

        if s == 1:
            return {'stage': 1, 'spanmask_p': 0.6, 'worddrop_p': 0.3, 'force_mask_k': 4}
        elif s == 2:
            return {'stage': 2, 'spanmask_p': 0.8, 'worddrop_p': 0.3, 'force_mask_k': 6}
        else:
            return {'stage': 3, 'spanmask_p': 1.0, 'worddrop_p': 1.0, 'force_mask_k': 9999}  # 完全に消すイメージ
            # ※ Stage3 は後述の compose でゼロ埋めにしてリーク遮断します


    def make_spanmask_worddrop(self, B, L, spanmask_p=0.6, worddrop_p=0.3, force_mask_k=4, device='cpu'):
        """
        戻り値: keep_mask (B, L) bool
        True=そのトークンの埋め込みを可視(使う)、False=隠す(ゼロ埋め等)
        - SpanMask: 連続区間をまとめて隠す
        - WordDrop: 個別にランダムで隠す
        - 先頭Kトークンは必ずマスク (prefix 依存を強制)
        # 例: L=10, spanmask_p=0.6, worddrop_p=0.3, force_mask_k=2
        original:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 全てTrue
        force_mask:[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 先頭2つマスク
        spanmask:  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 連続区間マスク
        worddrop:  [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]  # 個別ドロップ
        final:     [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]  # 最終結果
        """
        keep = torch.ones(B, L, dtype=torch.bool, device=device)

        # 先頭K は必ず隠す
        K = min(force_mask_k, L)
        if K > 0:
            keep[:, :K] = False

        # SpanMask: 目標マスク率に近づくまでランダムに区間を消す
        target_mask = int(L * spanmask_p)
        for b in range(B):
            masked = (~keep[b]).sum().item()
            # span の平均長（適当な実用値）
            mean_span = max(3, L // 10)
            # 過剰に回さない
            tries = 0
            while masked < target_mask and tries < 50:
                span = max(1, int(torch.normal(float(mean_span), float(mean_span*0.5), size=(1,)).abs().item()))
                start = torch.randint(0, max(1, L - span + 1), (1,)).item()
                keep[b, start:start+span] = False
                masked = (~keep[b]).sum().item()
                tries += 1

        # WordDrop: 残りの可視トークンにも個別ドロップ
        drop = torch.rand(B, L, device=device) < worddrop_p
        keep = keep & (~drop)

        # 先頭Kは二重保障で必ず False
        if K > 0:
            keep[:, :K] = False

        # 極端に全部Falseだと学習不安定なので「最低1つはTrue」を保証
        for b in range(B):
            if not keep[b].any():
                keep[b, torch.randint(0, L, (1,)).item()] = True

        return keep  # bool(B,L)


    def lail_loss(self, prefix1, prefix2, input_ids, stage_cfg):
        """
        LAIL損失: CE + 0.3·KL(T=2.5)
        - prefix1: CNN末からのprefix (B, Tp, d_llm)
        - prefix2: RNN中段からのprefix (B, Tp, d_llm)
        - 両方でLLM forwardし、CE損失の平均 + KL divergenceを計算
        """
        # prefix1でLLM forward
        inputs_embeds1, attn1, labels1 = self.compose_llm_inputs(prefix1, input_ids, stage_cfg)
        out1 = llm_model(inputs_embeds=inputs_embeds1, attention_mask=attn1, labels=labels1)

        # prefix2でLLM forward
        inputs_embeds2, attn2, labels2 = self.compose_llm_inputs(prefix2, input_ids, stage_cfg)
        out2 = llm_model(inputs_embeds=inputs_embeds2, attention_mask=attn2, labels=labels2)

        # CE損失（平均）
        ce_loss = (out1.loss + out2.loss) / 2.0

        # KL損失（温度T=2.5）- 数値安定化版
        T = 2.5
        logits1 = out1.logits  # (B, L, vocab_size)
        logits2 = out2.logits

        # ★ 数値安定化: logitsをクリップしてオーバーフローを防ぐ
        logits1 = torch.clamp(logits1, -100, 100)
        logits2 = torch.clamp(logits2, -100, 100)

        # 温度スケーリングしてKL divergence計算
        p1 = F.softmax(logits1 / T, dim=-1)
        log_p2 = F.log_softmax(logits2 / T, dim=-1)

        # KL(p1 || p2) を計算（prefix部分は無視するためlabels=-100の位置をマスク）
        # 簡易版: 全体でKLを計算（厳密にはlabels!=-100の位置のみ）
        kl_raw = F.kl_div(log_p2, p1, reduction='batchmean') * (T ** 2)  # 温度補正

        # ★ 数値安定化: KL divergenceをクリップ（異常に高い値を防ぐ）
        kl_loss = torch.clamp(kl_raw, 0.0, 10.0)

        # ★ NaN/Infチェック
        if not torch.isfinite(ce_loss):
            print(f"[WARNING] CE loss is not finite: {ce_loss.item()}, setting to 0")
            ce_loss = torch.tensor(0.0, device=ce_loss.device)
        if not torch.isfinite(kl_loss):
            print(f"[WARNING] KL loss is not finite (raw={kl_raw.item()}), setting to 0")
            kl_loss = torch.tensor(0.0, device=kl_loss.device)

        return ce_loss + 0.3 * kl_loss, float(ce_loss.item()), float(kl_loss.item())

    def compose_llm_inputs(self, prefix, input_ids, stage_cfg):
        """
        - prefix: (B, Tp, d_llm)
        - input_ids: (B, L)
        戻り値: inputs_embeds (B, Tp+L, d_llm), attn_full (B, Tp+L), labels_full (B, Tp+L)
        ルール:
        Stage1/2:
            - 可視トークンは埋め込みを使う
            - マスクされたトークンはゼロ埋めベクトルに置換
        Stage3:
            - テキスト部分は全ゼロ埋め（Prefix-only 相当）
        共通:
            - prefix 部分の labels は -100
            - テキスト部分の labels は input_ids（HFが内部で1トークン右にずらす）
        """
        B, Tp, D = prefix.shape
        tok = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
        # 埋め込みを一括で取得（LLMデバイス/精度）
        tok_emb = llm_model.get_input_embeddings()(input_ids.to(self.device_llm))   # (B, L, D)

        # Stageごとの可視マスク
        if stage_cfg['stage'] == 1:
            keep = self.make_spanmask_worddrop(
                B, input_ids.size(1),
                spanmask_p=stage_cfg['spanmask_p'],
                worddrop_p=stage_cfg['worddrop_p'],
                force_mask_k=stage_cfg['force_mask_k'],
                device=self.device_llm
            )
        elif stage_cfg['stage'] == 2:
            keep = self.make_spanmask_worddrop(
                B, input_ids.size(1),
                spanmask_p=stage_cfg['spanmask_p'],
                worddrop_p=stage_cfg['worddrop_p'],
                force_mask_k=stage_cfg['force_mask_k'],
                device=self.device_llm
            )
        else:
            # Stage3: テキスト全部マスク扱い
            keep = torch.zeros(B, input_ids.size(1), dtype=torch.bool, device=self.device_llm)

        # マスクされた位置はゼロ埋めベクトルに置換（リーク遮断）
        zero_vec = torch.zeros(1, 1, D, device=self.device_llm, dtype=tok_emb.dtype)
        masked_tok_emb = torch.where(
            keep.unsqueeze(-1), tok_emb, zero_vec.expand(B, input_ids.size(1), D)
        )

        # inputs_embeds = [prefix, masked_tok_emb]
        inputs_embeds = torch.cat([prefix.to(self.device_llm, dtype=tok_emb.dtype), masked_tok_emb], dim=1)

        # attention は全部1で OK（ゼロ埋め位置にも attend 可能：学習しやすい）
        attn_full = torch.ones(B, Tp + input_ids.size(1), dtype=torch.long, device=self.device_llm)

        # labels: prefix 部分は ignore(-100)、テキスト部分は元の input_ids
        labels_full = torch.full((B, Tp, ), fill_value=-100, device=self.device_llm, dtype=input_ids.dtype)
        labels_full = torch.cat([labels_full, input_ids.to(self.device_llm)], dim=1)

        return inputs_embeds, attn_full, labels_full


    def train(self, epoch):

        alpha_llm_base = getattr(self.config.train, 'alpha_llm', 0.1)
        # 追加：LLM間引きの間隔（kステップに1回）
        k = int(getattr(self.config.train, 'llm_interval', 8))

        # ★ LLM損失のウォームアップ
        llm_warmup_epochs = getattr(self.config.train, 'llm_warmup_epochs', 0)
        if epoch <= llm_warmup_epochs:
            # ウォームアップ期間中は徐々に増やす
            alpha_llm = alpha_llm_base * (epoch / max(llm_warmup_epochs, 1))
        else:
            alpha_llm = alpha_llm_base

        # ★ LLM間引き確認ログ
        print(f"[LLM Interval] LLM損失は {k} ステップに1回計算されます (alpha_llm={alpha_llm:.4f}, base={alpha_llm_base})")

        config = self.config
        device = config.device

        self.net.train()

        t = tqdm.tqdm(self.loaders['train'])
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, (img, transcr) in enumerate(t):
            self.optimizer.zero_grad()

            img = img.to(device)

            if config.arch.head_type == "both":
                output, aux_output = self.net(img)
            else:
                output = self.net(img)

            # 出力 shape を主張（CTC は (T,N,C)）
            assert output.dim() == 3, output.shape
            T, N, C = output.shape  # output.size(0) が T、size(1) が N

            # labels / lengths は CPU の long に固定
            labels = torch.as_tensor([self.classes['c2i'][c] for c in ''.join(transcr)],
                                    dtype=torch.long, device='cpu')
            label_lens = torch.as_tensor([len(t) for t in transcr],
                                        dtype=torch.long, device='cpu')
            act_lens  = torch.as_tensor([T] * N, dtype=torch.long, device='cpu')

            # 整合チェック（ズレてたらここで分かる）
            assert labels.numel() == int(label_lens.sum())
            assert (label_lens > 0).all() and (act_lens > 0).all()
            assert (label_lens <= act_lens).all()

            # ★ NaN/Inf検出と詳細ログ
            if not torch.isfinite(output).all():
                print(f"\n[ERROR] output has NaN/Inf detected!")
                print(f"  - output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
                print(f"  - NaN count: {torch.isnan(output).sum().item()}")
                print(f"  - Inf count: {torch.isinf(output).sum().item()}")
                print(f"  - Skipping this batch to continue training...")
                continue  # このバッチをスキップして次へ

            if labels.numel() > 0:  # 空でなければ値域チェック
                assert int(labels.min()) >= 0 and int(labels.max()) < C

            # log-softmax は CUDA 上で (T,N,C)
            logp = F.log_softmax(output, dim=2).contiguous()

            # ★ cuDNN を避ける“安全CTC”を使用
            loss_val = self.ctc_loss_safe(logp, labels, act_lens, label_lens)

            ctc_only = loss_val.clone()
            
            if config.arch.head_type == "both":
                loss_val += 0.1 * self.ctc_loss(aux_output, labels, act_lens, label_lens)



            llm_loss = torch.tensor(0.0, device=device)  # 表示用
            ce_loss_val = 0.0
            kl_loss_val = 0.0

            # ---- LLM 間引き：kステップに1回だけ実行 ----
            use_llm = (alpha_llm > 0) and ((iter_idx % k) == 0)

            if use_llm and (config.arch.head_type == "both") and \
               (self.net.top.llm_prefix1 is not None) and (self.net.top.llm_prefix2 is not None):
                # ★ LLM計算開始ログ
                import time
                llm_start = time.time()

                # ★ Connector×2からprefix1とprefix2を取得
                prefix1 = self.net.top.llm_prefix1.to(self.device_llm, dtype=self.lm_dtype)
                prefix2 = self.net.top.llm_prefix2.to(self.device_llm, dtype=self.lm_dtype)

                # トークナイズ → ids を LLM デバイスへ
                tok = llm_tokenizer(list(transcr), return_tensors='pt', padding=True, add_special_tokens=True)
                input_ids = tok['input_ids'].to(self.device_llm)

                # Stageごとの設定を取得
                stage_cfg = self.build_stage_settings()

                # ★ LAIL損失を計算（CE + 0.3·KL）
                llm_loss, ce_loss_val, kl_loss_val = self.lail_loss(prefix1, prefix2, input_ids, stage_cfg)

                loss_val = loss_val + (alpha_llm * k) * llm_loss.to(device)

                # ★ LLM計算時間ログ
                llm_time = time.time() - llm_start
                if iter_idx % (k * 5) == 0:  # 40ステップ(8*5)に1回詳細ログ
                    print(f"\n[LLM Computed] iter={iter_idx}, time={llm_time:.2f}s, CE={ce_loss_val:.2f}, KL={kl_loss_val:.2f}")
                
            tloss_val = float(loss_val.item())
            
        
            loss_val.backward()
            self.tb.log_train_step(
                ctc_only=float(ctc_only.item()),
                llm_loss=float(llm_loss.item()),
                total=float(loss_val.item()),
                stage=self.stage_monitor.stage,
                ratio_ema=self.stage_monitor.ratio_ema,
                grad_ema=self.stage_monitor.grad_ema,
                ce_loss=ce_loss_val,
                kl_loss=kl_loss_val
            )

            # ★ Connector×2の勾配を計算
            conn_grad = None
            s, n = 0.0, 0
            if hasattr(self.net.top, 'connector1') and (self.net.top.connector1 is not None):
                for p in self.net.top.connector1.parameters():
                    if p.grad is not None:
                        s += p.grad.detach().abs().mean().item(); n += 1
            if hasattr(self.net.top, 'connector2') and (self.net.top.connector2 is not None):
                for p in self.net.top.connector2.parameters():
                    if p.grad is not None:
                        s += p.grad.detach().abs().mean().item(); n += 1
            conn_grad = (s / n) if n > 0 else None

            llm_loss_val = float(llm_loss.item())


            self.stage_monitor.update_batch(
                ctc_loss=float(ctc_only.item()),
                llm_loss=llm_loss_val,
                connector_grad_mean=conn_grad,
                alpha_llm=alpha_llm,
                k=k
            )

            # ★ Gradient clipping を追加（NaN/Inf防止）
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)

            self.optimizer.step()

            # ★ tqdm表示更新（LLM間引きを考慮）
            if config.arch.head_type == "both" and alpha_llm > 0:
                if use_llm and (iter_idx % k) == 0:
                    # LLM計算したステップ: CE/KL損失も表示
                    t.set_postfix_str('loss: {:.2f} | llm: {:.2f} (CE: {:.2f}, KL: {:.2f}) [LLM@{}/{}]'.format(
                        tloss_val, float(llm_loss.item()), ce_loss_val, kl_loss_val, iter_idx, k))
                else:
                    # LLM計算しないステップ
                    t.set_postfix_str('loss: {:.2f} | llm: skip'.format(tloss_val))
            else:
                t.set_postfix_str('loss: {:.2f}'.format(tloss_val))

        self.sample_decoding()
        dep_ratio = self.dependency_test_small(tset='val', max_B=4)
        self.stage_monitor.on_eval_dependency(dep_ratio, cer=None, wer=None)

        # ★ StageMonitor デバッグログ
        print(f"\n[StageMonitor Debug] Epoch {epoch}")
        print(f"  Stage: {self.stage_monitor.stage}, Passes: {self.stage_monitor.passes}/{self.stage_monitor.passes_needed}")
        print(f"  dep_ratio: {dep_ratio:.3f} (need: {self.stage_monitor.dep12 if self.stage_monitor.stage==1 else self.stage_monitor.dep23})")
        print(f"  ratio_ema: {self.stage_monitor.ratio_ema:.3f} (band: {self.stage_monitor.ratio_lo:.2f}-{self.stage_monitor.ratio_hi:.2f})")
        print(f"  grad_ema: {self.stage_monitor.grad_ema:.5f} (band: {self.stage_monitor.grad_lo:.5f}-{self.stage_monitor.grad_hi:.5f})")

        new_stage, changed = self.stage_monitor.maybe_advance(epoch)
        self.tb.log_dep_ratio(epoch, dep_ratio)
        if changed:
            print(f"[StageMonitor] Stage advanced to {new_stage} at epoch {epoch}")
        
    
    @torch.no_grad()
    def dependency_test_small(self, tset='val', max_B=4):
        device = self.config.device
        loader = self.loaders['val'] if tset=='val' else self.loaders['test']
        imgs, transcrs = next(iter(loader))
        imgs = imgs[:max_B].to(device)
        transcrs = transcrs[:max_B]

        self.net.eval()
        o = self.net(imgs)
        if self.config.arch.head_type == 'both':
            _ = o[0]  # llm_prefix1, llm_prefix2 を更新するために forward だけ

        # ★ Connector2（RNN中段）を使って依存度テスト（より情報量が多い）
        if hasattr(self.net.top, 'llm_prefix2') and (self.net.top.llm_prefix2 is not None):
            prefix = self.net.top.llm_prefix2.to(self.device_llm, dtype=self.lm_dtype)
        else:
            # フォールバック: prefix1を使う
            prefix = self.net.top.llm_prefix1.to(self.device_llm, dtype=self.lm_dtype)

        tok = llm_tokenizer(list(transcrs), return_tensors='pt', padding=True, add_special_tokens=True)
        input_ids = tok['input_ids'].to(self.device_llm)

        stage_cfg = self.build_stage_settings()
        inputs_embeds, attn_full, labels_full = self.compose_llm_inputs(prefix, input_ids, stage_cfg)
        out = llm_model(inputs_embeds=inputs_embeds, attention_mask=attn_full, labels=labels_full)
        L_base = float(out.loss.item())

        zeros = torch.zeros_like(prefix)
        inputs_embeds_np = torch.cat([zeros, inputs_embeds[:, prefix.size(1):]], dim=1)
        out_np = llm_model(inputs_embeds=inputs_embeds_np, attention_mask=attn_full, labels=labels_full)
        L_noprefix = float(out_np.loss.item())

        self.net.train()
        return L_noprefix / max(L_base, 1e-8)

    def test(self, epoch, tset='test'):

        config = self.config
        device = config.device

        self.net.eval()

        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        for (imgs, transcrs) in tqdm.tqdm(loader):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = self.net(imgs)
            # if o tuple keep only the first element
            if config.arch.head_type == 'both':
                o = o[0]
            
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy()
            if tdecs.ndim == 1:            # バッチサイズが1でも (1, T) にする
                tdecs = tdecs[None, :]
            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        self.net.train()
        self.tb.log_eval(epoch, cer=cer_score, wer=wer_score, split=tset)
        

    def save(self, epoch):
        print('####################### Saving model at epoch {} #######################'.format(epoch))
        if not os.path.exists(self.config.model.save_dir):
            os.makedirs(self.config.model.save_dir)

        torch.save(self.net.cpu().state_dict(), self.config.model.save_dir + '/{}.pt'.format(epoch))
        self.net.to(self.config.device)


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    max_epochs = config.train.num_epochs

    htr_trainer = HTRTrainer(config)

    cnt = 1
    print('Training Started!')
    htr_trainer.test(0, 'test')
    for epoch in range(1, max_epochs + 1):

        htr_trainer.train(epoch)
        htr_trainer.scheduler.step()

        if epoch == 1:
            htr_trainer.save(epoch)


        # save and evaluate the current model
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            htr_trainer.test(epoch, 'val')
            htr_trainer.test(epoch, 'test')
            
            
    
    if not os.path.exists(config.model.save_dir):
            os.makedirs(config.model.save_dir)
    torch.save(htr_trainer.net.cpu().state_dict(), config.model.save_dir + '/{}'.format(config.save))
    htr_trainer.tb.close()
