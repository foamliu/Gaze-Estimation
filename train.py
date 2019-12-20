import warnings

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import device, grad_clip, print_freq, num_workers
from data_gen import GazeEstimationDataset
from models import GazeEstimationModel, GazeEstimationMobile
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger

warnings.simplefilter(action='ignore', category=FutureWarning)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model = GazeEstimationModel()
        model = GazeEstimationMobile()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay, nesterov=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model = nn.DataParallel(model)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.SmoothL1Loss()

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(GazeEstimationDataset('train'), batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(GazeEstimationDataset('val'), batch_size=args.batch_size, shuffle=False,
                                             num_workers=num_workers)

    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)

        lr = optimizer.param_groups[0]['lr']
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model/learning_rate', lr, epoch)

        # One epoch's validation
        val_loss = valid(val_loader=val_loader,
                         model=model,
                         criterion=criterion,
                         logger=logger)

        writer.add_scalar('model/val_loss', val_loss, epoch)

        scheduler.step(epoch)

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    l_losses = AverageMeter()
    p_losses = AverageMeter()

    # Batches
    for i, (img, lbl_look_vec, lbl_pupil_size) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        lbl_look_vec = lbl_look_vec.float().to(device)  # [N, 3]
        lbl_pupil_size = lbl_pupil_size.float().to(device)  # [N, 1]

        # Forward prop.
        out_look_vec, out_pupil_size = model(img)  # embedding => [N, 3]

        # Calculate loss
        loss1 = criterion(out_look_vec, lbl_look_vec)
        loss2 = criterion(out_pupil_size, lbl_pupil_size)
        loss2 = loss2  # / 20
        loss = loss1 + loss2

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item() * 1000, img.size(0))
        l_losses.update(loss1.item() * 1000, img.size(0))
        p_losses.update(loss2.item() * 1000, img.size(0))

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                        'Look Vec Loss {l_loss.val:.5f} ({l_loss.avg:.5f})\t'
                        'Pupil Size Loss {p_loss.val:.5f} ({p_loss.avg:.5f})'.format(epoch, i,
                                                                                     len(train_loader),
                                                                                     loss=losses,
                                                                                     l_loss=l_losses,
                                                                                     p_loss=p_losses))

    return l_losses.avg + p_losses.avg


def valid(val_loader, model, criterion, logger):
    model.eval()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    l_losses = AverageMeter()
    p_losses = AverageMeter()

    # Batches
    for (img, lbl_look_vec, lbl_pupil_size) in tqdm(val_loader):
        # Move to GPU, if available
        img = img.to(device)
        lbl_look_vec = lbl_look_vec.float().to(device)  # [N, 3]
        lbl_pupil_size = lbl_pupil_size.float().to(device)  # [N, 1]

        # Forward prop.
        with torch.no_grad():
            out_look_vec, out_pupil_size = model(img)  # embedding => [N, 3]

        # Calculate loss
        loss1 = criterion(out_look_vec, lbl_look_vec)
        loss2 = criterion(out_pupil_size, lbl_pupil_size)
        loss2 = loss2  # / 20
        loss = loss1 + loss2

        # Keep track of metrics
        losses.update(loss.item() * 1000, img.size(0))
        l_losses.update(loss1.item() * 1000, img.size(0))
        p_losses.update(loss2.item() * 1000, img.size(0))

    # Print status
    status = 'Validation\t' \
             'Loss {loss.avg:.5f}\t' \
             'Look Vec Loss {l_loss.avg:.5f}\t' \
             'Pupil Size Loss {p_loss.avg:.5f}\n'.format(
        loss=losses,
        l_loss=l_losses,
        p_loss=p_losses)
    logger.info(status)

    return l_losses.avg + p_losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
