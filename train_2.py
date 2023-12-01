from utils_2 import *
from model_2 import *
from data import *

from pytorch_lightning.loggers import CSVLogger


class SegFormer_B0(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(3, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8),
            MixTransformerEncoderLayer(64, 160, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=2, num_heads=5, expansion_factor=4),
            MixTransformerEncoderLayer(160, 256, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=1, num_heads=8, expansion_factor=4)
        ])
        self.decoder = MLPDecoder([32, 64, 160, 256], 256, (64, 64), 4)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, images):
        embeds = [images]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return self.decoder(embeds[1:])
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=6e-5)
        return optimizer
    
    def miou(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same shape")
    
        batch_size, _, _ = prediction.shape
        iou_sum = 0
        
        for i in range(batch_size):
            prediction_cls = prediction[i]
            target_cls = target[i]
            
            intersection = np.logical_and(prediction_cls, target_cls).sum()
            union = np.logical_or(prediction_cls, target_cls).sum()
            
            if union == 0:
                continue
            else:
                iou = intersection / union
            iou_sum += iou
        
        mean_iou = iou_sum / batch_size
        return mean_iou

    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.forward(images)
        loss = self.loss(predictions, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.forward(images)

        # target_np = one_hot(targets)
        target_np = targets.cpu().numpy()
        prediction_np = predictions.detach().cpu().numpy()
        prediction_np = np.argmax(prediction_np, axis=1)
        
        # prediction_np = np.where(prediction_np > threshold, 1, 0)
        miou = self.miou(prediction_np, target_np)
        self.log('miou', miou, prog_bar=True)


if __name__ == '__main__':

    train_dataset = ADE20K('data', 'train', download=True, transforms=simple_ade20k_transforms)
    val_dataset = ADE20K('data', 'val', transforms=simple_ade20k_transforms)

    valid_filename = 'valid_indices.pkl'
    valid_indices = load_indices(valid_filename)
    if valid_indices is None:
        valid_indices = trim_dataset(val_dataset)
        save_indices(valid_indices, valid_filename)

    train_filename = 'train_indices.pkl'
    train_indices = load_indices(train_filename)
    if train_indices is None:
        train_indices = trim_dataset(train_dataset)
        save_indices(train_indices, train_filename)

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, valid_indices)

    model = SegFormer_B0()
    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(val_dataset, batch_size=16)

    csv_logger = CSVLogger(save_dir='logs/', name='my_model')
    trainer = pl.Trainer(devices=1, accelerator=device.type, max_epochs=20, logger=csv_logger)
    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), 'model.pth')