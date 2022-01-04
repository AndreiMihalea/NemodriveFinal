import os
from models.resnet import *
from models.simple import *
from models.pilot import *
from util.dataset import UPBDataset
from torch.utils.data import DataLoader


nbins=401
model = RESNET(no_outputs=nbins, use_roi='none').cpu()
# model = Simple(no_outputs=nbins).cpu()
# model= PilotNet(no_outputs=nbins).cpu()


dataset_dir = os.path.join('/mnt/storage/workspace/andreim/nemodrive/UPB_dataset_robert/dataset_paper/', "pose_dataset")
train_dataset = UPBDataset(dataset_dir, train=True, augm=True, scale=32.8, roi='gt_soft')

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=False,
    num_workers=2 // 2
)


batch = next(iter(train_dataloader))
yhat = model(batch) # Give dummy batch to forward().


input_names = ['Input']
output_names = ['Output']
torch.onnx.export(model, batch, 'onnx_models/pilot.onnx', input_names=input_names, output_names=output_names)
