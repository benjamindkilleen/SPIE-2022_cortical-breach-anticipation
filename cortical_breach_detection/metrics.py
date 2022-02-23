import logging

import torch
import torchmetrics
from torch import nn

log = logging.getLogger(__name__)


class AggregateMetrics(nn.Module):
    def __init__(self, *metrics):
        super().__init__()
        self.trues = []
        self.preds = []
        self.metrics = metrics

    def forward(self, y_pred, y_true, mode="train"):
        self.trues.append(y_true.cpu())
        self.preds.append(y_pred.detach().cpu())

        trues = torch.cat(self.trues, dim=0)
        preds = torch.cat(self.preds, dim=0)

        results = {}
        if len(trues.unique()) != 1:
            for metric in self.metrics:
                try:
                    results.update(
                        {
                            f"{mode}/{metric.__class__.__name__.lower()}": metric(
                                preds.cpu(), trues.cpu()
                            )
                        }
                    )
                except ValueError:
                    log.error(f"{metric.__class__.__name__} metric failed.")
                    continue
        return results

    def reset(self):
        self.trues = []
        self.preds = []
