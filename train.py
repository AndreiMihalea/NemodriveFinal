import random
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import argparse
import json
import os
import pandas as pd

from models.resnet import *
from models.simple import *
from models.pilot import *

from util.vis import *
from util.io import *
from util.early import *
from util.dataset import UPBDataset

from tqdm import tqdm
from tensorboardX import SummaryWriter
from typing import Tuple


def get_args() -> argparse.Namespace:
    """
    Parses console arguments

    Returns
    -------
    Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-4, help="final learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--step_size", type=int, default=5, help="scheduler learning rate step size")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer available: adam, rmsprop")
    parser.add_argument("--log_interval", type=int, default=50, help="number of batches to log after")
    parser.add_argument("--vis_interval", type=int, default=500, help="number of batches to visualize after")
    parser.add_argument("--save_interval", type=int, default=1, help="number of epoch to save the model after")
    parser.add_argument("--log_dir", type=str, default="./logs", help="logging directory")
    parser.add_argument("--vis_dir", type=str, default="./snapshots", help="visualize directory")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="dataset directory")
    parser.add_argument("--num_workers", type=int, default=10, help="number of workers for dataloader")
    parser.add_argument("--num_vis", type=int, default=10, help="number of visualizations")
    parser.add_argument("--use_augm", action="store_true", help="use augmentation dataset")
    parser.add_argument("--use_synth", action="store_true", help="use a syntethic dataset for testing")
    parser.add_argument("--use_speed", action="store_true", help="append speed to nvidia model")
    parser.add_argument("--use_balance", action="store_true", help="balance training dataset")
    parser.add_argument("--use_pose", action="store_true", help="use pose estimation dataset")
    parser.add_argument("--use_scheduler", action="store_true", help="use linear lr scheduler")
    parser.add_argument("--load_model", type=str, help="checkpoint name", default=None)
    parser.add_argument("--patience", type=int, default=6, help="early stopping patience")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay optimizer")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--scale", type=float, default=1.0, help="scaling factor for the radius")
    args = parser.parse_args()
    return args


def set_seed(seed: int = 0):
    """
    Sets seed for reproducible results

    Parameters
    ----------
    seed
        seed for reproducible results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)


def compile(args: argparse.Namespace) -> Tuple:
    """
    Defines the model, criterion, optimizer and scheduler
    according to the parsed arguments

    Parameters
    ----------
    args
        parsed arguments

    Returns
    -------
    Tuple consisting of model, optimizer, criterion
    """
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    nbins=401
    model = RESNET(no_outputs=nbins).to(device)
    # model = Simple(no_outputs=nbins).to(device)
    # model= PilotNet(no_outputs=nbins).to(device)

    # define criterion
    criterion = nn.KLDivLoss(reduction="batchmean")

    # define optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.step_size, verbose=True)
    # scheduler = None
    return model, optimizer, criterion, scheduler


def get_dataset(args: argparse.Namespace) -> Tuple[Tuple, Tuple]:
    # define data loaders
    dataset_dir = os.path.join(args.dataset_dir, "pose_dataset" if args.use_pose else "gt_dataset")
    train_dataset = UPBDataset(dataset_dir, train=True, augm=args.use_augm, scale=args.scale)
    test_dataset = UPBDataset(dataset_dir, train=False, scale=args.scale)
    synth_test_dataset = UPBDataset(dataset_dir, train=False, synth=args.use_synth, scale=args.scale)

    # balanced training dataset
    if args.use_balance:
        weights_file = "weights.csv"
        weights = pd.read_csv(os.path.join(dataset_dir, weights_file)).to_numpy().reshape(-1)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers // 2
        )
    
    # define normal test dataset
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers // 2
    )
    
    # define synthetic test dataset
    # this dataset contains the test images
    # that were perspectively augmented with
    # a predefine tx, ry
    synth_test_dataloader = DataLoader(
        synth_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers // 2
    )

    return (train_dataset, train_dataloader), (test_dataset, test_dataloader),\
            (synth_test_dataset, synth_test_dataloader)


def run_epoch(dataloader, epoch, train_flag=True, synth_flag=False):
    global rloss

    # total loss
    total_loss = 0.0

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, data in enumerate(dataloader):
        if train_flag:
            optimizer.zero_grad()

        # pass data through model
        for key in data:
            data[key] = data[key].to(device)

        if train_flag:
            turning_logits = model(data)
        else:
            with torch.no_grad():
                turning_logits = model(data)

        # compute steering loss
        log_softmax_output = F.log_softmax(turning_logits, dim=1)
        loss = criterion(log_softmax_output, data["turning_pmf"])

        # gradient step
        if train_flag:
            loss.backward()
            optimizer.step()

        # update running losses
        if train_flag:
            rloss = loss.item() if rloss is None else rloss * 0.99 + 0.01 * loss.item()
        
        # update total loss
        batch_size = data['turning'].shape[0]
        total_loss += loss.item() * batch_size
        
        # print statistics
        if train_flag and i % args.log_interval == 0:
            print("Epoch: %d, Batch: %d, Running loss: %.4f" % (epoch, i, rloss))
            index = epoch * (len(train_dataset) // args.batch_size) + i
            writer.add_scalar("loss", loss.item(), index)            
            writer.flush()

        # visualization
        if i % args.vis_interval == 0:
            num_vis = min(args.num_vis, args.batch_size)
            softmax_output = F.softmax(turning_logits, dim=1)
            
            if train_flag:
                folder = "imgs_train" 
            else:
                folder = "imgs_synth_test" if synth_flag else "imgs_test"

            path = os.path.join(args.vis_dir, experiment, folder, "epoch:%d.batch:%d.png" % (epoch, i))
            visualisation(
                img=data["img"][:num_vis],
                course=data["turning_pmf"][:num_vis],
                softmax_output=softmax_output[:num_vis], 
                num_vis=num_vis, 
                path=path
            )
    return total_loss


def linear_lr_scheduler(optimizer, init_lr, final_lr, epoch, max_epoch):
    """
    Linear learning rate scheduler.
    Decreasease learning rate linear with epoch.

    Parameters
    ----------
    optimizer
        parameters optimizer
    init_lr
        initial learning rate
    final_lr
        final learning rate
    epoch
        current epoch
    max_epoch
        final epoch
    """
    
    alpha = epoch / max_epoch
    lr = init_lr * (1 - alpha) + final_lr * alpha

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    # get parsed arguments
    args = get_args()

    # set seed
    set_seed(args.seed)

    # create necessary directories
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(args.vis_dir):
        os.mkdir(args.vis_dir)

    # define experiment name
    experiment = str(len(os.listdir(args.vis_dir))).zfill(5)

    if not os.path.exists(os.path.join(args.vis_dir, experiment)):
        os.makedirs(os.path.join(args.vis_dir, experiment))
        os.makedirs(os.path.join(args.vis_dir, experiment, "imgs_train"))
        os.makedirs(os.path.join(args.vis_dir, experiment, "imgs_test"))
        os.makedirs(os.path.join(args.vis_dir, experiment, "imgs_synth_test"))
        os.makedirs(os.path.join(args.vis_dir, experiment, "ckpts"))

    # save args as json in the experiment folder
    path = os.path.join(args.vis_dir, experiment, "args.txt")
    with open(path, 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)

    # define writer & running loss
    writer = SummaryWriter(os.path.join(args.log_dir, experiment))

    # get model, criterion, optimizer and scheduler
    model, optimizer, criterion, scheduler = compile(args)

    # get dataset
    (train_dataset, train_dataloader), (test_dataset, test_dataloader), (synth_test_dataset, synth_test_dataloader) = get_dataset(args)

    # define training variables
    start_epoch = 0
    best_score, rloss = None, None

    # load model if necessary
    if args.load_model:
        ckpt_name = os.path.join(args.vis_dir, args.load_model, "ckpts", "default.pth")
        start_epoch = load_ckpt(
                ckpt_name=ckpt_name,
                models=[('model', model)], 
                optimizers=[('optimizer', optimizer)],
                schedulers=[('scheduler', scheduler)],
                rlosses=[('rloss', rloss)], 
                best_scores=[('best_score', best_score)]
        )

    # define early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        # learning rate scheduler
        #if args.use_scheduler:
        #    linear_lr_scheduler(optimizer, args.lr, args.final_lr, epoch, args.num_epochs)
        print("current learning rate: %f" % (optimizer.param_groups[0]['lr']))

        # train step        
        model.train()
        train_loss = run_epoch(train_dataloader, epoch=epoch, train_flag=True)
        train_loss /= len(train_dataset)
        
        # test step
        model.eval()
        test_loss =  run_epoch(test_dataloader, epoch=epoch, train_flag=False)
        test_loss /= len(test_dataset)

        # test step on the synthetic dataset
        if args.use_synth:
            model.eval()
            synth_test_loss = run_epoch(synth_test_dataloader, epoch=epoch, train_flag=False, synth_flag=True)
            synth_test_loss /= len(synth_test_dataset)
            print("Real Test Loss: %.4f, Synth Test Loss: %.4f" % (test_loss, synth_test_loss))
            
            # add the syntethic loss to the test loss
            test_loss = 0.5 * (test_loss + synth_test_loss)

        # log
        print("Epoch: %d, Train Loss: %.4f, Test Loss: %.4f" % (epoch, train_loss, test_loss))
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.flush()

        if True: # epoch % args.save_interval == 0 and (best_score is None or best_score > test_loss):
            best_score = test_loss
            name = "epoch: %d; tloss: %.2f; vloss: %.2f.pth" % (epoch, train_loss, test_loss)
            ckpt_name = os.path.join(args.vis_dir, experiment, "ckpts", name)
            save_ckpt(
                ckpt_name, 
                models=[('model', model)], 
                optimizers=[('optimizer', optimizer)],
                schedulers=[('scheduler', scheduler)],
                rlosses=[('rloss', rloss)], 
                best_scores=[('best_score', best_score)],
                n_iter=epoch + 1
            )
            ckpt_name = os.path.join(args.vis_dir, experiment, "ckpts", "default.pth")
            save_ckpt(
                ckpt_name,
                models=[('model', model)],
                optimizers=[('optimizer', optimizer)],
                schedulers=[('scheduler', scheduler)],
                rlosses=[('rloss', rloss)],
                best_scores=[('best_score', best_score)],
                n_iter=epoch + 1
            )
            print("Model saved!")

        # learning rate scheduler step
        scheduler.step()
        # scheduler.step(test_loss)
        
        # early stopping
        early_stopping(test_loss)
        if early_stopping.early_stop:
            break

    writer.close()
