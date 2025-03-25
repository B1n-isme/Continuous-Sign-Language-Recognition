# lightningmodule: model architecture
## init: wrap all nn.Module architecture
- self.save_hyperparameters() to ckpt (not preferable with large custom module)
- pass module directly when instantiate and load_from_checkpoint
```python
def __init__(self, encoder, decoder, *args, **kwargs):
    super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # load pretrained
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        # Optionally, save hyperparameters for later use (excluding large modules if needed)
        self.save_hyperparameters(ignore=['model'])
```

## training_step
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    self.log("train_loss", loss)
    return loss
```
## optimizer configuration
```python
def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
```
## Validation_step
```python
def validation_step(self, batch, batch_idx):
    loss, acc = self._shared_eval_step(batch, batch_idx)
    metrics = {"val_acc": acc, "val_loss": loss}
    self.log_dict(metrics)
    return metrics
```
## test_step
```python
def test_step(self, batch, batch_idx):
    loss, acc = self._shared_eval_step(batch, batch_idx)
    metrics = {"test_acc": acc, "test_loss": loss}
    self.log_dict(metrics)
    return metrics
```
## predict_step
```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    x, y = batch
    y_hat = self.model(x)
    return y_hat
```
# data_loader

# Trainer: lightningmodule + data_loader
- auto save ckpt
- default_root_dir: 
- Early stopping:
		early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
trainer = Trainer(callbacks=[early_stop_callback])

- Use model: 
Simply load: model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt", module=module)
	a.  predict
	model.eval()
	y_hat = model(x)
	b. Access hyperparameter
	print(model.learning_rate)

Resume train state
model = LitModel()
trainer = Trainer()
# automatically restores model, epoch, step, LR schedulers, etc…
trainer.fit(model, ckpt_path="path/to/your/checkpoint.ckpt")



