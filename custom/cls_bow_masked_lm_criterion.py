import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import pandas as pd

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.masked_lm import MaskedLmLoss


@register_criterion("cls_bow_mlm")
class CLSBowMaskedLmLoss(MaskedLmLoss):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.vocab_size = len(task.dictionary)
        self.mask_idx = task.dictionary.symbols.index("<mask>")
        self.cls_loss_alpha = getattr(args, "cls_loss_alpha", 1.0)
        self.grad_norm_interval = getattr(args, "grad_norm_interval", 100)
        self.cls_metrics_interval = getattr(args, "cls_metrics_interval", 100)
        self.cls_ones_weight = getattr(args, "cls_ones_weight", 20.0)
        self.scores_path = getattr(args, "scores_path", None)
        self.scores_ids = None
        self.cls_pred_dir = Path(self.args.save_dir).parent / "cls_pred"
        self.cls_pred_dir.mkdir(exist_ok=True, parents=True)

        if self.scores_path:
            scores_df = pd.read_csv(self.scores_path)
            self.scores_ids = scores_df.sort_values(by="scores", ascending=False).head(100)["keys"].tolist()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--cls-loss-alpha", default=1.0, type=float,
                            help="weight in sum MLM loss and CLS loss")
        parser.add_argument("--grad-norm-interval", default=100, type=int,
                            help="frequency for computing CLS and MLM grad norms")
        parser.add_argument("--cls-metrics-interval", default=100, type=int,
                            help="frequency for computing CLS related metrics")
        parser.add_argument("--cls-ones-weight", default=20.0, type=float,
                            help="weight of 1s in CLS loss")
        parser.add_argument("--scores-path", default="", type=str,
                            help="Path to DataFrame with token scores")

    @overrides
    def forward(self, model: nn.Module,
                sample: OrderedDict,
                reduce: bool = True):
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        if not sample_size:
            masked_tokens = None

        src_tokens = sample["net_input"]["src_tokens"]
        device = src_tokens.device
        mlm_targets = sample["target"]
        cls_targets = torch.zeros((src_tokens.size(0), self.vocab_size), device=device).float()
        not_corrupted_tokens = src_tokens.clone()

        mlm_logits, extra = model(**sample['net_input'], masked_tokens=masked_tokens)

        if sample_size:
            not_corrupted_tokens[masked_tokens] = mlm_targets[masked_tokens]
            mlm_targets = mlm_targets[masked_tokens]

        row_mask = torch.arange(src_tokens.size(0)).view(-1, 1).repeat(1, not_corrupted_tokens.size(1))
        cls_targets[row_mask, not_corrupted_tokens] = 1.0

        # Make sure pad index set to zero in CLS targets
        cls_targets[:, self.padding_idx] = 0.0

        mlm_loss = F.nll_loss(
            F.log_softmax(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                dim=-1
            ),
            mlm_targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        cls_logits = extra["cls_bow_logits"]
        pos_weight = self.cls_ones_weight * torch.ones(cls_targets.size(1), device=cls_targets.device)
        cls_loss = F.binary_cross_entropy_with_logits(
            F.logsigmoid(
                cls_logits
            ),
            cls_targets,
            reduction="mean",
            pos_weight=pos_weight
        )

        loss = mlm_loss + self.cls_loss_alpha * cls_loss
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            "cls_loss": utils.item(cls_loss.data) if reduce else cls_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        coef = self.args.distributed_world_size / float(sample_size) if sample_size else 1.0
        track_gnorms = (self.task.num_updates + 1) % self.grad_norm_interval == 0 and mlm_logits.grad_fn is not None
        if self.task.num_updates == 1 or track_gnorms:
            cls_grads = torch.autograd.grad(outputs=[self.cls_loss_alpha * cls_loss],
                                            inputs=list(model.parameters()),
                                            only_inputs=True,
                                            retain_graph=True,
                                            allow_unused=True)
            mlm_grads = torch.autograd.grad(outputs=[mlm_loss],
                                            inputs=list(model.parameters()),
                                            only_inputs=True,
                                            retain_graph=True,
                                            allow_unused=True)
            cls_grad_norm = torch.stack([(grad * coef).norm(2) for grad in cls_grads if grad is not None]).norm(2)
            mlm_grad_norm = torch.stack([(grad * coef).norm(2) for grad in mlm_grads if grad is not None]).norm(2)

            logging_output["cls_gnorm"] = utils.item(cls_grad_norm.data) if reduce else cls_grad_norm.data
            logging_output["mlm_gnorm"] = utils.item(mlm_grad_norm.data) if reduce else mlm_grad_norm.data

        track_metrics = (self.task.num_updates + 1) % self.cls_metrics_interval == 0 or mlm_logits.grad_fn is None
        if self.task.num_updates == 1 or track_metrics:
            cls_pred = (cls_logits > 0.0).float()

            clf_pred_path = self.cls_pred_dir / f"{self.task.num_updates}_upd.out"
            with clf_pred_path.open("a") as fp:
                for batch_idx in range(cls_pred.size(0)):
                    if batch_idx > int(25. / self.args.update_freq[0]) + 1:
                        break
                    fp.write("corr=" + " ".join([str(int(x)) for x in src_tokens[batch_idx, 1:-1].tolist()]) + "\n")
                    fp.write("orig=" + " ".join([str(int(x)) for x in not_corrupted_tokens[batch_idx, 1:-1].tolist()]) + "\n")
                    fp.write("preds=" + "".join([str(x)[0] for x in cls_pred[batch_idx, :].tolist()]) + "\n")

            train_cls_acc = (cls_targets == cls_pred).sum().float() / (cls_targets.size(0) * cls_targets.size(1))
            logging_output["cls_acc"] = utils.item(train_cls_acc.data) if reduce else train_cls_acc.data

            y_true = cls_targets.cpu().numpy().ravel()
            y_pred = cls_pred.cpu().numpy().ravel()
            pr_scores, rc_scores, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
            logging_output["cls_prec_0"] = pr_scores[0]
            logging_output["cls_prec_1"] = pr_scores[1]
            logging_output["cls_rec_0"] = rc_scores[0]
            logging_output["cls_rec_1"] = rc_scores[1]
            logging_output["cls_f1_0"] = f1_scores[0]
            logging_output["cls_f1_1"] = f1_scores[1]

            if self.scores_path:
                y_true = cls_targets[:, self.scores_ids].cpu().numpy().ravel()
                y_pred = cls_pred[:, self.scores_ids].cpu().numpy().ravel()
                pr_scores, rc_scores, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
                logging_output["score_prec_0"] = pr_scores[0]
                logging_output["score_prec_1"] = pr_scores[1]
                logging_output["score_rec_0"] = rc_scores[0]
                logging_output["score_rec_1"] = rc_scores[1]
                logging_output["score_f1_0"] = f1_scores[0]
                logging_output["score_f1_1"] = f1_scores[1]

            mask_for_masked = (src_tokens == self.mask_idx)
            masks_per_sample = mask_for_masked.sum(dim=1)
            row_ids = torch.cat([
                torch.tensor([i]).repeat(masks_per_sample[i])
                for i in torch.arange(cls_targets.size(0))
            ])

            # Class 0 - tokens that should not be
            # Class 1 - tokens for coping
            # Class 2 - masked tokens
            targets = cls_targets.clone()
            targets[row_ids, not_corrupted_tokens[mask_for_masked]] = 2.0
            cls_pred[row_ids, not_corrupted_tokens[mask_for_masked]] = 2.0
            cls_pred[(cls_pred == 2.0) & (cls_logits <= 0.0)] = 0.0

            mask_for_replaced = (src_tokens != sample["target"]) & \
                                (src_tokens != self.mask_idx) & \
                                (sample["target"] != self.padding_idx)
            replaces_per_sample = mask_for_replaced.sum(dim=1)
            row_ids = torch.cat([
                torch.tensor([i]).repeat(replaces_per_sample[i])
                for i in torch.arange(cls_targets.size(0))
            ])
            # Class 3 - tokens in not corrupted input that was replaced
            targets[row_ids, not_corrupted_tokens[mask_for_replaced]] = 3.0
            cls_pred[row_ids, not_corrupted_tokens[mask_for_replaced]] = 3.0
            cls_pred[(cls_pred == 3.0) & (cls_logits <= 0.0)] = 0.0

            # Class 4 - tokens that replace original tokens
            targets[row_ids, src_tokens[mask_for_replaced]] = 4.0
            cls_pred[row_ids, src_tokens[mask_for_replaced]] = 4.0
            cls_pred[(cls_pred == 4.0) & (cls_logits > 0.0)] = 1.0

            y_true = targets.cpu().numpy().ravel()
            y_pred = cls_pred.cpu().numpy().ravel()
            pr_scores, rc_scores, f1_scores, _ = precision_recall_fscore_support(y_true,
                                                                                 y_pred,
                                                                                 average=None,
                                                                                 labels=[0, 1, 2, 3, 4])
            logging_output["precision_0"] = pr_scores[0]
            logging_output["recall_0"] = rc_scores[0]
            logging_output["f_score_0"] = f1_scores[0]

            logging_output["precision_1"] = pr_scores[1]
            logging_output["recall_1"] = rc_scores[1]
            logging_output["f_score_1"] = f1_scores[1]

            logging_output["precision_2"] = pr_scores[2]
            logging_output["recall_2"] = rc_scores[2]
            logging_output["f_score_2"] = f1_scores[2]

            logging_output["precision_3"] = pr_scores[3]
            logging_output["recall_3"] = rc_scores[3]
            logging_output["f_score_3"] = f1_scores[3]

            logging_output["precision_4"] = pr_scores[4]
            logging_output["recall_4"] = rc_scores[4]
            logging_output["f_score_4"] = f1_scores[4]

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        num_logs = len(logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)
        cls_loss = sum(log.get("cls_loss", 0) for log in logging_outputs)

        agg_output = {
            "nll_loss": nll_loss / sample_size / math.log(2) if ntokens > 0 else 0.,
            "cls_loss": cls_loss / num_logs / math.log(2) if nsentences > 0 else 0.,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        agg_output["loss"] = agg_output["nll_loss"] + agg_output["cls_loss"]
        if logging_outputs[0].get("cls_acc", None) is not None:
            agg_output["cls_acc"] = sum(log.get("cls_acc", 0) for log in logging_outputs) / num_logs
            agg_output["cls_prec_0"] = sum(log.get("cls_prec_0", 0) for log in logging_outputs) / num_logs
            agg_output["cls_prec_1"] = sum(log.get("cls_prec_1", 0) for log in logging_outputs) / num_logs
            agg_output["cls_rec_0"] = sum(log.get("cls_rec_0", 0) for log in logging_outputs) / num_logs
            agg_output["cls_rec_1"] = sum(log.get("cls_rec_1", 0) for log in logging_outputs) / num_logs
            agg_output["cls_f1_0"] = sum(log.get("cls_f1_0", 0) for log in logging_outputs) / num_logs
            agg_output["cls_f1_1"] = sum(log.get("cls_f1_1", 0) for log in logging_outputs) / num_logs

        if logging_outputs[0].get("score_prec_0", None) is not None:
            agg_output["score_prec_0"] = sum(log.get("score_prec_0", 0) for log in logging_outputs) / num_logs
            agg_output["score_prec_1"] = sum(log.get("score_prec_1", 0) for log in logging_outputs) / num_logs
            agg_output["score_rec_0"] = sum(log.get("score_rec_0", 0) for log in logging_outputs) / num_logs
            agg_output["score_rec_1"] = sum(log.get("score_rec_1", 0) for log in logging_outputs) / num_logs
            agg_output["score_f1_0"] = sum(log.get("score_f1_0", 0) for log in logging_outputs) / num_logs
            agg_output["score_f1_1"] = sum(log.get("score_f1_1", 0) for log in logging_outputs) / num_logs

        if logging_outputs[0].get("precision_0", None) is not None:
            agg_output["precision_0"] = sum(log.get("precision_0", 0) for log in logging_outputs) / num_logs
            agg_output["recall_0"] = sum(log.get("recall_0", 0) for log in logging_outputs) / num_logs
            agg_output["f_score_0"] = sum(log.get("f_score_0", 0) for log in logging_outputs) / num_logs
            agg_output["precision_1"] = sum(log.get("precision_1", 0) for log in logging_outputs) / num_logs
            agg_output["recall_1"] = sum(log.get("recall_1", 0) for log in logging_outputs) / num_logs
            agg_output["f_score_1"] = sum(log.get("f_score_1", 0) for log in logging_outputs) / num_logs
            agg_output["precision_2"] = sum(log.get("precision_2", 0) for log in logging_outputs) / num_logs
            agg_output["recall_2"] = sum(log.get("recall_2", 0) for log in logging_outputs) / num_logs
            agg_output["f_score_2"] = sum(log.get("f_score_2", 0) for log in logging_outputs) / num_logs
            agg_output["precision_3"] = sum(log.get("precision_3", 0) for log in logging_outputs) / num_logs
            agg_output["recall_3"] = sum(log.get("recall_3", 0) for log in logging_outputs) / num_logs
            agg_output["f_score_3"] = sum(log.get("f_score_3", 0) for log in logging_outputs) / num_logs
            agg_output["precision_4"] = sum(log.get("precision_4", 0) for log in logging_outputs) / num_logs
            agg_output["recall_4"] = sum(log.get("recall_4", 0) for log in logging_outputs) / num_logs
            agg_output["f_score_4"] = sum(log.get("f_score_4", 0) for log in logging_outputs) / num_logs

        if logging_outputs[0].get("cls_gnorm", None) is not None:
            agg_output["cls_gnorm"] = sum(log.get("cls_gnorm", 0) for log in logging_outputs)
            agg_output["mlm_gnorm"] = sum(log.get("mlm_gnorm", 0) for log in logging_outputs)

        return agg_output
