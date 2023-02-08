import torch
import pytorch_lightning as pl
import evaluate
from transformers import SegformerForSemanticSegmentation
from torch.nn import CrossEntropyLoss
#from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
#from utils.utils import VisualizeSampleCallback
from data_handler.data import CamvidDataset


class SegFormerFineTuned(pl.LightningModule):
  def __init__(self, id2label, 
               train_dl, 
               val_dl,
               metrics_interval,
               class_weights,
               model_path="nvidia/segformer-b0-finetuned-ade-512-512"):
    
    super(SegFormerFineTuned, self).__init__()
    self.id2label = id2label
    self.metrics_interval = metrics_interval
    self.train_dl = train_dl
    self.val_dl = val_dl
    self.weights = class_weights
    self.model_path = model_path

    self.num_classes = len(id2label.keys())
    self.label2id = {v:k for k,v in self.id2label.items()}
    
    self.model = SegformerForSemanticSegmentation.from_pretrained(
        self.model_path, 
        return_dict=False, 
        num_labels=self.num_classes,
        id2label=self.id2label,
        label2id=self.label2id,
        ignore_mismatched_sizes=True,
    )
    
    self.train_mean_iou = evaluate.load("mean_iou") 
    self.val_mean_iou = evaluate.load("mean_iou") 
    self.test_mean_iou = evaluate.load("mean_iou")
    
    # Save the hyper-parameters
    # with the checkpoints
    self.save_hyperparameters()
  
  def forward(self, images, masks):
    outputs = self.model(pixel_values=images)
    return (outputs)

  def training_step(self, batch, num_batch):
    images, masks = batch['pixel_values'], batch['labels']

    # Forward pass    
    
    predictions = self(images,masks)[0]
    
    # upsample the predictions 
    # from size (H/4,W/4) -> (H,W)
    predictions = torch.nn.functional.interpolate(
            predictions, 
            size=masks.shape[-2:], 
            mode="nearest-exact", 
            align_corners=False
        )
  
    weighted_loss = CrossEntropyLoss(weight=self.weights,ignore_index=255)
    loss = weighted_loss(predictions,masks)
    
    predictions = predictions.argmax(dim=1)
    
    
    # Evaluate the model
    self.train_mean_iou.add_batch(
            predictions= predictions.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
    if num_batch % self.metrics_interval == 0:

        metrics = self.train_mean_iou.compute(
            num_labels=self.num_classes, 
            ignore_index=255, 
            reduce_labels=False,
        )
        
        metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return(metrics)
    else:
        return({'loss': loss})
  
  def validation_step(self, batch, num_batch):
    images, masks = batch['pixel_values'], batch['labels']

    # Forward pass    
    
    predictions = self(images,masks)[0]
    
    # up-samples the predictions 
    # from size (H/4,W/4) -> (H,W)
    predictions = torch.nn.functional.interpolate(
            predictions, 
            size=masks.shape[-2:], 
            mode="nearest-exact", 
            align_corners=False
        )
    weighted_loss = CrossEntropyLoss(weight=self.weights,ignore_index=255)
    loss = weighted_loss(predictions,masks)
    predictions = predictions.argmax(dim=1)


    # Evaluate the model
    self.val_mean_iou.add_batch(
            predictions= predictions.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
    
    return({'val_loss': loss})

  def validation_epoch_end(self,outputs):
    metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        
    avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    val_mean_iou = metrics["mean_iou"]
    val_mean_accuracy = metrics["mean_accuracy"]
    
    metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
    for k,v in metrics.items():
        self.log(k,v)

    return metrics
  
  def configure_optimizers(self):
    return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
  def train_dataloader(self):
    return self.train_dl
  
  def val_dataloader(self):
    return self.val_dl

def train_model(train_dataloader,
                val_dataloader,
                class_weights,
                id2label,
                hf_model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                ckpt_path='/checkpoints/',
                accelerator_mode='gpu',
                devices=1,
                max_epochs=300,
                log_every_n_steps=8,
                last_ckpt_path=None,
                resume=False
            ):
    
    if accelerator_mode == "gpu":
        model = SegFormerFineTuned(
            id2label, 
            train_dl=train_dataloader, 
            val_dl=val_dataloader, 
            metrics_interval=log_every_n_steps,
            class_weights=torch.Tensor(class_weights).cuda(),
            model_path=hf_model_name
        )
    else:
        model = SegFormerFineTuned(
            id2label, 
            train_dl=train_dataloader, 
            val_dl=val_dataloader, 
            metrics_interval=log_every_n_steps,
            class_weights=torch.Tensor(class_weights),
            model_path=hf_model_name
        )

    # Callback to stop when the model stops improving
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="min",
    )
    # monitor the evolution of training and validation metrics
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    # Callback to see a prediction sample by the end of the training
    #visualize_callback = VisualizeSampleCallback()

    trainer = pl.Trainer(
        default_root_dir=ckpt_path,
        accelerator=accelerator_mode,
        devices=devices, 
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=max_epochs,
        log_every_n_steps= log_every_n_steps,
        val_check_interval=len(train_dataloader),
    )

    if resume and last_ckpt_path:
      trainer.fit(model,ckpt_path=last_ckpt_path)
    else:
      trainer.fit(model)
    
    return trainer