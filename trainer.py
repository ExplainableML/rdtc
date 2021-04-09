import os
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, dataloaders, optimizer, scheduler, num_epochs,
                 device, log_path):

        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.log_path = log_path

        self.logger = SummaryWriter(self.log_path)

    def train(self):
        self.run_model('train', self.dataloaders['train'], epoch_end=self.num_epochs)

    def test(self):
        self.run_model('test', self.dataloaders['test'])

    def run_model(self, mode, data_laoder, epoch_start=0, epoch_end=1):
        max_cls_accuracy = 0
        max_cls_agg_accuracy = 0

        if mode == 'train':
            self.model.train()
            ctx = nullcontext()
        else:
            self.model.eval()
            ctx = torch.no_grad()

        self.model.reset_stats()

        # Train the Model
        epoch_iter = range(epoch_start, epoch_end)
        if mode == 'train':
            epoch_iter = tqdm(epoch_iter, desc=f'{mode.upper()} Epochs')
        for epoch in epoch_iter:
            stats = self.init_stats()
            tqdm_step = tqdm(data_laoder, total=len(data_laoder))
            for data in tqdm_step:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                if mode == 'train':
                    self.optimizer.zero_grad()

                with ctx:
                    classification, loss = self.model(images, labels)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()

                self.update_stats(stats, classification, labels, loss)

                lr = self.optimizer.param_groups[0]['lr'] if mode == 'train' else None
                self.set_tqdm_description(tqdm_step, f'Epoch {epoch+1}', mode,
                                          stats['total_loss'] / stats['total'], lr)

            self.calc_stats(stats)
            tqdm.write(f"Accuracy ({mode}), Top1: {100*stats['correct_1_mean_cls'][-1]:.2f}, Top5: {100*stats['correct_5_mean_cls'][-1]:.2f}")

            if mode == 'train':
                self.scheduler.step()
                val_stats = self.run_model('val', self.dataloaders['val'],
                                           epoch_start=epoch, epoch_end=epoch+1)

                self.log_stats('train', epoch, stats)
                self.log_stats('val', epoch, val_stats)

                val_cls_accuracy = val_stats['correct_1_mean_cls'][-1]
                val_cls_agg_accuracy = val_stats['correct_1_mean_cls'].sum()
                if val_cls_accuracy > max_cls_accuracy:
                    max_cls_accuracy = val_cls_accuracy
                    self.save_model('best_clsacc')
                if val_cls_agg_accuracy > max_cls_agg_accuracy:
                    max_cls_agg_accuracy = val_cls_agg_accuracy
                    self.save_model('best_clsacc_agg')

                self.save_model('latest')

            if mode == 'test':
                self.print_test_stats(stats)

        if mode != 'train':
            self.model.train()

        return stats

    def init_stats(self):
        stats = {}
        n_stats = self.model.max_iters
        stats['correct_1'] = np.zeros((n_stats, self.model.num_classes))
        stats['correct_5'] = np.zeros((n_stats, self.model.num_classes))
        stats['total'] = 0
        stats['total_cnt'] = np.zeros((1, self.model.num_classes))
        stats['total_loss'] = 0
        return stats

    def update_stats(self, stats, classification, labels, loss):
        # Collect stats
        stats['total_loss'] += loss.item() * labels.size(0)
        stats['total'] += labels.size(0)

        for k in range(len(classification)):
            ctopk, target_cnt = self.topk_correct(classification[k].data, labels, (1, 5))
            c1, c5 = ctopk
            stats['correct_1'][k] += c1
            stats['correct_5'][k] += c5

        stats['total_cnt'][0] += target_cnt

        return stats

    def calc_stats(self, stats):
        total = stats['total']
        total_cnt = stats['total_cnt']
        stats['correct_1_mean'] = stats['correct_1'].sum(axis=1) / total
        stats['correct_5_mean'] = stats['correct_5'].sum(axis=1) / total
        stats['total_loss_mean'] = stats['total_loss'] / total
        nonzeros = total_cnt.nonzero()[1]
        total_cnt = total_cnt[:, nonzeros]
        stats['correct_1_mean_cls'] = (stats['correct_1'][:, nonzeros] / total_cnt).mean(axis=1)
        stats['correct_5_mean_cls'] = (stats['correct_5'][:, nonzeros] / total_cnt).mean(axis=1)

        stats['unique_attr'] = self.model.get_unique_attributes()
        stats['pruning_ratio'] = self.model.get_pruning_ratio(total)
        if self.model.attribute_coef > 0.:
            stats['attr_acc'] = self.model.get_attr_acc(total)
        else:
            stats['attr_acc'] = None

    def print_test_stats(self, stats, column_size=10):
        disp_attr = stats['attr_acc'] is not None
        row_format = f'{{:>{column_size}}}' * (4 + int(disp_attr))
        column_names = ['Depth', 'Accuracy', 'UniqAttr', 'Pruned']
        columns = [range(1, len(stats['correct_1_mean_cls'])+1),
                   stats['correct_1_mean_cls'],
                   stats['unique_attr'],
                   stats['pruning_ratio']]
        if disp_attr:
            column_names.insert(2, 'AttrAcc')
            columns.insert(2, stats['attr_acc'])

        print(row_format.format(*column_names))
        #for d, acc, attr, uniq, pruned in zip(*columns):
        for row in zip(*columns):
            row = [np.around(c, 5) if isinstance(c, float) else c for c in row]

            print(row_format.format(*row))
            #print(row_format.format(d, np.around(acc, 5), np.around(attr, 5),
            #                        uniq, np.around(pruned, 5)))

    def save_model(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.log_path, '{}.pth'.format(name)))

    def topk_correct(self, output, target, topk=(1,)):
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        target_masks = []
        target_cnt = []
        for i in range(self.model.num_classes):
            target_masks.append((target == i).unsqueeze(0))
            target_cnt.append(target_masks[i].sum().item())

        res = []
        for k in topk:
            correct_k = [(correct[:k] * tm).reshape(-1).float().sum(0, keepdim=True).item() for tm in target_masks]
            res.append(np.array(correct_k))
        return res, np.array(target_cnt)

    def log_stats(self, phase, epoch, epoch_stats):
        for k in range(len(epoch_stats['correct_1_mean_cls'])):
            d = k+1
            self.logger.add_scalar(f'metric/Top1Accuracy{d}/{phase}', epoch_stats['correct_1_mean'][k], epoch)
            self.logger.add_scalar(f'metric/Top5Accuracy{d}/{phase}', epoch_stats['correct_5_mean'][k], epoch)
            self.logger.add_scalar(f'metric/Top1MeanClassAccuracy{d}/{phase}', epoch_stats['correct_1_mean_cls'][k], epoch)
            self.logger.add_scalar(f'metric/Top5MeanClassAccuracy{d}/{phase}', epoch_stats['correct_5_mean_cls'][k], epoch)
            self.logger.add_scalar(f'metric/UniqueAttributes{d}/{phase}', epoch_stats['unique_attr'][k], epoch)
            if epoch_stats['attr_acc'] is not None:
                self.logger.add_scalar(f'metric/AttributeAccuracy{d}/{phase}', epoch_stats['attr_acc'][k], epoch)
        self.logger.add_scalar('loss/'+phase, epoch_stats['total_loss_mean'], epoch)

    def set_tqdm_description(self, tqdm_iterator, prefix, mode, loss, lr):
        tqdm_str = f'{mode.upper()} {prefix}'
        tqdm_str += f'; loss: {loss:.5f}'
        tqdm_str += f'; lr: {lr:.5f}' if lr is not None else ''

        tqdm_iterator.set_description(tqdm_str)
