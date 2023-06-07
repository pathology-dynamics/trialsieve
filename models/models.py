import torch
import lightning.pytorch as pl
from transformers import AutoModelForTokenClassification
from lightning.pytorch.loggers import WandbLogger


class NERModel(pl.LightningModule):
    def __init__(self, hf_basemodel):
        super().__init__()
        self.basemodel = AutoModelForTokenClassification.from_pretrained(hf_basemodel)
        self.training_step_preds = []
        self.val_step_preds = []
        self.test_step_preds = []

        

    def forward(
        self,
        input_ids,
        attention_mask,
        token_labels,
    ):
        return self.basemodel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=token_labels,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output_dict = self(input_ids, attention_mask, labels)
        loss = output_dict["loss"]
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        logits = output_dict['logits']
        print(logits.shape)

        output_preds = torch.argmax(logits, dim=2)
        self.training_step_outputs.append(output_preds)
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.stack(self.training_step_outputs)
        # do something with all preds
        ...
        self.training_step_outputs.clear()  # free memory

    def _shared_eval_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output_dict = self(input_ids, attention_mask, labels)
        loss = output_dict["loss"]
        return loss
    
    def validation_step(self, batch, batch_idx):

        self.validation_step_outputs.append(pred)
        return pred


    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs)
        # do something with all preds
        ...
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="MNIST")
    trainer = Trainer(logger=wandb_logger)
    wandb_logger = WandbLogger(log_model="all")
