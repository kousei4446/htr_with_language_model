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
from utils.logger import HTRLogger, LLMLossTracker

from models import HTRNet
from utils.transforms import aug_transforms

import torch.nn.functional as F

from utils.metrics import CER, WER
import re, torch

def migrate_rec_to_both(state: dict) -> dict:
    """Stage1の top.rec.* を Stage2の top.rec1.* / top.recN.* に写し替え。"""
    out = dict(state)
    pat = re.compile(r"^top\.rec\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)(?:(_reverse))?$")
    for k, v in list(state.items()):
        m = pat.match(k)
        if not m:
            continue
        kind, lvl, rev = m.group(1), int(m.group(2)), (m.group(3) or "")
        newk = f"top.rec1.{kind}_l0{rev}" if lvl == 0 else f"top.recN.{kind}_l{lvl-1}{rev}"
        out.setdefault(newk, v)  # 既にあれば触らない
    return out

def init_from_stage1(net: torch.nn.Module, ckpt_path: str, *, verbose: bool = True):
    """Stage1 ckptで Stage2モデルを初期化（必要なら rec→rec1/recN に自動マップ）。"""
    print(f"[init] load: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("state_dict", obj.get("model", obj))

    # DataParallel 'module.' 剥がし
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # モデル側が rec1/recN なのに ckpt が rec.* ならキーをマップ
    msd = net.state_dict()
    need_both = any(k.startswith(("top.rec1.", "top.recN.")) for k in msd.keys())
    has_rec   = any(k.startswith("top.rec.") for k in state.keys())
    if need_both and has_rec:
        state = migrate_rec_to_both(state)

    # 形状一致だけ安全に流し込む
    safe = {k: v for k, v in state.items() if (k in msd) and (msd[k].shape == v.shape)}
    incompat = net.load_state_dict(safe, strict=False)

    if verbose:
        print(f"[init] loaded {len(safe)}/{len(msd)} params")
        if hasattr(incompat, "missing_keys"):
            print(f"[init] missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")
    return incompat

class HTRTrainer(nn.Module):
    def __init__(self, config):
        super(HTRTrainer, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()
        self.prepare_logger()


    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=aug_transforms)
        classes = train_set.character_classes
        print('# training lines ' + str(train_set.__len__()))

        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        print('# validation lines ' + str(val_set.__len__()))

        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        print('# testing lines ' + str(test_set.__len__()))

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

    def prepare_net(self):

        config = self.config

        device = config.device

        print('Preparing Net - Architectural elements:')
        print(config.arch)

        classes = self.classes['classes']

        # use_llm フラグを取得（デフォルト: True）
        use_llm = config.train.get('use_llm', True)

        net = HTRNet(config.arch, len(classes) + 1, use_llm=use_llm)
        
        if config.resume is not None:
            _ = init_from_stage1(net, config.resume) 
            
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /self.config.train.batch_size

        self.lail_loss = lambda llm_output: llm_output.loss / self.config.train.batch_size if(llm_output is not None and hasattr(llm_output, 'loss')) else torch.tensor(0.0,device=self.config.device)

    def prepare_optimizers(self):
        config = self.config
        optimizer = torch.optim.AdamW(self.net.parameters(), config.train.lr, weight_decay=0.00005)

        self.optimizer = optimizer

        max_epochs = config.train.num_epochs
        if config.train.scheduler == 'mstep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
        else:
            raise NotImplementedError('Alternative schedulers not implemented yet')

    def prepare_logger(self):
        """TensorBoardロガーを初期化"""
        self.logger = HTRLogger(config=self.config)

        # LLM使用状況を表示
        use_llm = self.config.train.get('use_llm', True)
        if self.config.arch.head_type == "both":
            if use_llm:
                llm_ratio = self.config.train.get('llm_sample_ratio', 0.125)
                print(f'LLM Learning: ENABLED (sample_ratio={llm_ratio:.1%})')
            else:
                print('LLM Learning: DISABLED (using CNN shortcut only)')

        # LLM損失追跡用
        llm_ratio = self.config.train.get('llm_sample_ratio', 0.125)
        self.llm_tracker = LLMLossTracker(llm_sample_ratio=llm_ratio)

    def decode(self, tdec, tdict, blank_id=0):
        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
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


    def train(self, epoch):

        config = self.config
        device = config.device

        self.net.train()

        t = tqdm.tqdm(self.loaders['train'])
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, (img, transcr) in enumerate(t):
            self.optimizer.zero_grad()

            img = img.to(device)

            # labels を先に定義（全サンプル用）
            labels = torch.IntTensor([self.classes['c2i'][c] for c in ''.join(transcr)])
            label_lens = torch.IntTensor([len(t) for t in transcr])

            # LLM使用フラグを取得
            use_llm = config.train.get('use_llm', True)

            if config.arch.head_type == "both":
                if use_llm:
                    # LLM有効: 毎バッチで1/8のサンプルをランダム選択
                    batch_size = img.size(0)
                    llm_ratio = config.train.get('llm_sample_ratio', 0.125)
                    llm_batch_size = max(1, int(batch_size * llm_ratio))

                    # ランダムインデックス選択
                    indices = torch.randperm(batch_size, device='cpu')[:llm_batch_size]
                    img_llm = img[indices]
                    transcr_llm = [transcr[i] for i in indices]

                    # モデル呼び出し（全サンプル + LLM用サンプル）
                    output, aux_output, llm_output = self.net(
                        img, img_llm=img_llm, transcr_llm=transcr_llm
                    )
                else:
                    # LLM無効: CNN shortcut のみ使用
                    output, aux_output, llm_output = self.net(
                        img, img_llm=None, transcr_llm=None
                    )
            else:
                output = self.net(img)
                aux_output, llm_output = None, None

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)])

            # CTC損失計算
            ctc_loss_val = self.ctc_loss(output, labels, act_lens, label_lens)
            loss_val = ctc_loss_val

            # 個別の損失を記録用に保存
            aux_loss_val = None
            llm_loss_val = None

            if config.arch.head_type == "both":
                # 補助損失（CNN shortcut）- head_type="both" なら常に計算
                aux_loss_val = self.ctc_loss(aux_output, labels, act_lens, label_lens)
                loss_val += 0.1 * aux_loss_val

                # LLM損失（use_llm=true の場合のみ計算）
                if use_llm:
                    llm_loss_raw = self.lail_loss(llm_output)
                    if llm_loss_raw.item() > 0:
                        llm_weight = 1.0 / llm_ratio
                        llm_loss_val = llm_loss_raw * llm_weight
                        loss_val += 0.1 * llm_loss_val
                        # LLM損失トラッカーに記録
                        self.llm_tracker.update(llm_loss_val.item())
                    else:
                        # LLM損失が計算されなかった
                        self.llm_tracker.update(None)

            tloss_val = loss_val.item()

            loss_val.backward()
            self.optimizer.step()

            # エポック平均計算用にバッファに保存（バッチごとのログは削除）
            self.logger.epoch_losses['total'].append(tloss_val)
            self.logger.epoch_losses['ctc'].append(ctc_loss_val.item())
            if aux_loss_val is not None:
                self.logger.epoch_losses['aux'].append(aux_loss_val.item())
            if llm_loss_val is not None:
                self.logger.epoch_losses['llm'].append(llm_loss_val.item())

            t.set_postfix(values='loss : {:.2f}'.format(tloss_val))

        # Epoch終了時の処理
        self.sample_decoding()

        # Epoch平均をログ記録
        self.logger.log_epoch_summary(epoch)

        # LLM損失の統計を表示（use_llm=true の場合のみ）
        if config.arch.head_type == "both" and use_llm:
            llm_stats = self.llm_tracker.get_stats()
            print(f'[Epoch {epoch}] LLM Stats: avg_loss={llm_stats["avg_loss"]:.4f}, '
                  f'computed_ratio={llm_stats["computation_ratio"]:.2%} '
                  f'(expected={llm_stats["expected_ratio"]:.2%})')
            self.llm_tracker.reset()
    
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
            
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()

            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                cer.update(dec_transcr, transcr)
                wer.update(dec_transcr, transcr)
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))

        # TensorBoardにログ記録
        self.logger.log_metrics(epoch, cer_score, wer_score, split=tset)

        self.net.train()

    def save(self, epoch):
        print('####################### Saving model at epoch {} #######################'.format(epoch))

        if not os.path.exists(config.model.save_dir):
            os.makedirs(config.model.save_dir)

        torch.save(self.net.cpu().state_dict(), config.model.save_dir + '/{}.pt'.format(epoch))

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

        # 学習率をログ記録
        current_lr = htr_trainer.optimizer.param_groups[0]['lr']
        htr_trainer.logger.log_learning_rate(epoch, current_lr)

        if epoch == 1:
            htr_trainer.save(epoch)

        # save and evaluate the current model
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            htr_trainer.test(epoch, 'val')
            htr_trainer.test(epoch, 'test')

    # save the final model

    if not os.path.exists(config.model.save_dir):
        os.makedirs(config.model.save_dir)
    torch.save(htr_trainer.net.cpu().state_dict(), config.model.save_dir + '/{}'.format(config.save))

    # TensorBoardロガーを閉じる
    htr_trainer.logger.close()
    print('Training completed!')
