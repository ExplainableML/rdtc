from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .cnn import get_cnn


class RDTC(nn.Module):
    def __init__(self, num_classes,
                 dataset, decision_size=2, max_iters=30, attribute_size=20,
                 attribute_mtx=None, attribute_coef=0., hidden_size=100,
                 tau_initial=5, tau_target=0.5, use_pretrained=False,
                 threshold=1.):
        super().__init__()
        self.num_classes = num_classes
        self.attribute_size = attribute_size
        self.attribute_mtx = attribute_mtx
        self.attribute_coef = attribute_coef if attribute_mtx is not None else 0.
        self.decision_size = decision_size
        self.tau_initial = tau_initial
        self.tau_target = tau_target
        self.max_iters = max_iters
        self.threshold = threshold
        self.stats = defaultdict(list)

        assert decision_size == 2 or (decision_size > 2 and self.attribute_coef == 0.), \
            'Attribute loss only supported for decision_size == 2'

        self.cnn, cnn_out_size = self.init_cnn(dataset, use_pretrained)
        self.init_network(hidden_size, decision_size, num_classes,
                          attribute_size, cnn_out_size)

        self.init_losses()

    def init_network(self, hidden_size, decision_size, num_classes,
                     attribute_size, cnn_out_size):
        assert decision_size > 1

        # LSTM initialization parameters
        self.init_h0 = nn.Parameter(
            torch.zeros(hidden_size).uniform_(-0.01, 0.01), requires_grad=True
        )
        self.init_c0 = nn.Parameter(
            torch.zeros(hidden_size).uniform_(-0.01, 0.01), requires_grad=True
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.classifier = nn.Sequential(
                nn.Linear(attribute_size * decision_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, num_classes)
        )

        self.question_mlp = nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, attribute_size)
        )

        self.attribute_mlp = nn.Sequential(
                nn.BatchNorm1d(cnn_out_size),
                nn.Linear(cnn_out_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, attribute_size * decision_size)
        )

        self.pre_lstm = nn.Sequential(
                nn.Linear(2 * attribute_size * decision_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size)
        )

        # Temperature parameters
        self.attribute_mlp.tau = nn.Parameter(torch.tensor([self.tau_initial],
                                              dtype=torch.float), requires_grad=True)
        self.question_mlp.tau = nn.Parameter(torch.tensor([self.tau_initial],
                                             dtype=torch.float), requires_grad=True)

    def init_losses(self):
        self.cls_loss = nn.CrossEntropyLoss()
        self.attr_loss = nn.BCEWithLogitsLoss()

    def init_cnn(self, dataset, use_pretrained):
        if use_pretrained:
            cnn_state_dict = torch.load('pretrained/{}_resnet152.pkl'.format(dataset))
            cnn, cnn_out_size = get_cnn(cnn_state_dict, freeze_weights=True)
        else:
            cnn, cnn_out_size = get_cnn()

        return cnn, cnn_out_size

    def get_initial_state(self, batch_size):
        h0 = self.init_h0.view(1, 1, -1).expand(-1, batch_size, -1)
        c0 = self.init_c0.view(1, 1, -1).expand(-1, batch_size, -1)
        state = (h0.contiguous(), c0.contiguous())
        return state

    def argmax(self, y_soft, dim):
        # Differentiable argmax
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        argmax = y_hard - y_soft.detach() + y_soft
        return argmax

    def reset_stats(self):
        self.unique_attributes = [set() for i in range(self.max_iters)]
        if self.attribute_coef > 0.:
            self.attr_pred_correct = np.zeros(self.max_iters)
        self.n_pruned = np.zeros(self.max_iters)

    def update_unique_attributes(self, attribute_idx):
        for iter, attr_idx in enumerate(attribute_idx):
            unique_attributes = attr_idx.unique().cpu().numpy()
            self.unique_attributes[iter] = self.unique_attributes[iter].union(unique_attributes)

    def get_unique_attributes(self):
        uniq_per_iter = []
        for i in range(self.max_iters):
            iter_set = self.unique_attributes[i]
            for j in range(i+1):
                if j == i:
                    continue
                iter_set = iter_set.union(self.unique_attributes[j])
            uniq_per_iter.append(len(iter_set))
        return uniq_per_iter

    def update_attr_preds(self, attr_correct):
        self.attr_pred_correct += attr_correct.cpu().numpy()

    def get_attr_acc(self, total_cnt):
        correct_cumsum = np.cumsum(self.attr_pred_correct)
        cnt_per_iter = (np.arange(self.max_iters) + 1) * total_cnt
        return correct_cumsum / cnt_per_iter

    def update_pruning_stats(self, threshold_masks):
        n_pruned = torch.stack(threshold_masks).sum(1).cpu().numpy()
        self.n_pruned += n_pruned

    def get_pruning_ratio(self, total_cnt):
        return self.n_pruned / total_cnt

    def apply_threshold(self, classification, threshold_mask, threshold_classification):

        above_thres = (F.softmax(classification, dim=1).max(dim=1)[0] > self.threshold)
        new_thres = (above_thres.int() - threshold_mask.int()).clamp(0., 1.).bool()
        threshold_classification[new_thres] = classification[new_thres]
        threshold_mask = threshold_mask | above_thres
        classification[threshold_mask] = threshold_classification[threshold_mask]

        return classification, threshold_mask, threshold_classification

    def compute_loss(self, labels, classifications, attribute_idx,
                     bin_attribute_logits=None):
        # Update attribute stats
        self.update_unique_attributes(attribute_idx)

        # Prepare dimensions
        n_iter = len(classifications)
        iter_labels = labels.repeat(n_iter)
        bin_attribute_logits = bin_attribute_logits.repeat(n_iter, 1)
        classifications = torch.cat(classifications, dim=0)
        attribute_idx = torch.cat(attribute_idx, dim=0)

        # RDT loss
        loss = (1. - self.attribute_coef) * self.cls_loss(classifications, iter_labels)

        # Attribute loss
        if self.attribute_coef > 0.:
            attribute_target = self.attribute_mtx[iter_labels, :].gather(1, attribute_idx.unsqueeze(1)).squeeze()
            attribute_pred = bin_attribute_logits.gather(1, attribute_idx.unsqueeze(1)).squeeze()
            loss += self.attribute_coef * self.attr_loss(attribute_pred,
                                                         attribute_target)

            # Update running attribute prediction accuracy
            attribute_pred_bin = (attribute_pred > 0.).long()
            self.update_attr_preds((attribute_pred_bin == attribute_target).view(n_iter, -1).sum(1))

        return loss

    def attribute_based_learner(self, images):
        img_feats = self.cnn(images)
        img_feats = img_feats.view(img_feats.size(0), -1)
        image_features = self.attribute_mlp(img_feats)

        attribute_logits = image_features.view(-1, self.decision_size)
        attributes_softmax = F.softmax(attribute_logits / self.attribute_mlp.tau, dim=1)
        attributes_hard = self.argmax(attributes_softmax, dim=1)
        image_features = attributes_hard.view(images.size(0), -1, self.decision_size)

        if self.attribute_coef > 0.:
            # Binary logits for attribute loss
            bin_attribute_logits = attribute_logits - attribute_logits[:, 1].unsqueeze(-1)
            bin_attribute_logits = bin_attribute_logits[:, 0].view(images.size(0), -1)
        else:
            bin_attribute_logits = None

        return image_features, bin_attribute_logits

    def make_decision(self, lstm_out, binary_features):
        # Perform categorical feature selection
        selection_logits = self.question_mlp(lstm_out)
        if self.training:
            hard_selection = F.gumbel_softmax(selection_logits, hard=True,
                                              tau=self.question_mlp.tau)
        else:
            hard_selection = self.argmax(selection_logits, dim=1)

        # Get single decision
        decision = (hard_selection.unsqueeze(2) * binary_features)
        decision = decision.view(-1, self.attribute_size * self.decision_size)

        # Index of decision
        attribute_idx = hard_selection.max(dim=1)[1]

        return decision, attribute_idx

    def run_rdt_iteration(self, binary_features, state, explicit_memory):
        lstm_out = state[0].squeeze(0)

        # Make binary decision
        decision, attribute_idx = self.make_decision(lstm_out, binary_features)

        if explicit_memory is None:
            explicit_memory = decision
        else:
            explicit_memory = (explicit_memory + decision).clamp(0., 1.)

        # Apply scaling similar to dropout scaling
        scaled_em = explicit_memory / explicit_memory.sum(dim=1).unsqueeze(1).detach()
        lstm_in = torch.cat((scaled_em, decision), dim=1)

        # Update LSTM state
        lstm_in = self.pre_lstm(lstm_in).unsqueeze(1)
        _, state = self.lstm(lstm_in, state)

        # Get current classification
        classification = self.classifier(scaled_em)

        return classification, state, explicit_memory, attribute_idx

    def recurrent_decision_tree(self, binary_features, labels):
        # Outputs for every iteration
        all_classifications = []
        all_attribute_idx = []

        if not self.training and self.threshold < 1.:
            all_threshold_masks = []

            threshold_mask = torch.zeros(len(labels), dtype=torch.bool).to(labels.device)
            threshold_classification = torch.zeros(labels.size(0), self.num_classes).to(labels.device)
        else:
            all_threshold_masks = None

        # Set initial state
        state = self.get_initial_state(binary_features.size(0))
        explicit_memory = None

        for j in range(self.max_iters):
            classification, state, explicit_memory, attribute_idx = self.run_rdt_iteration(
                binary_features, state, explicit_memory
            )

            if not self.training and self.threshold < 1.:
                all_threshold_masks.append(threshold_mask.clone())

                classification, threshold_mask, threshold_classification = self.apply_threshold(
                    classification, threshold_mask, threshold_classification
                )

            all_classifications.append(classification)
            all_attribute_idx.append(attribute_idx)

        return all_classifications, all_attribute_idx, all_threshold_masks

    def forward(self, images, labels):
        # Get categorical features once
        binary_features, bin_attribute_logits = self.attribute_based_learner(images)

        classification, attribute_idx, thres_mask = self.recurrent_decision_tree(
                binary_features, labels
        )

        if thres_mask is not None:
            self.update_pruning_stats(thres_mask)

        loss = self.compute_loss(labels, classification, attribute_idx,
                                 bin_attribute_logits)

        return classification, loss
