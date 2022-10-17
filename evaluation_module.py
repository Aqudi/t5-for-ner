import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import (
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl


class T5NerFineTuner(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path,
        weight_decay,
        learning_rate,
        adam_epsilon,
        warmup_steps,
        total_steps,
        **kwargs,
    ):
        super(T5NerFineTuner, self).__init__()
        self.save_hyperparameters()

        print(self.hparams)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )

    def forward(self, batch):
        return self.model(**batch)

    def _step(self, batch):
        outputs = self(batch)
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log_dict(
            {
                "avg_train_loss": avg_train_loss,
                "log": tensorboard_logs,
                "progress_bar": tensorboard_logs,
            }
        )

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.log_dict(
            {
                "avg_val_loss": avg_loss,
                "log": tensorboard_logs,
                "progress_bar": tensorboard_logs,
            }
        )

    def predict_step(self, batch, _):
        outputs = self(batch)
        # softmaxed = F.softmax(outputs.logits, dim=1)
        decode_outputs = torch.argmax(outputs[1], dim=-1).view(-1)
        return decode_outputs

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        return tqdm_dict
