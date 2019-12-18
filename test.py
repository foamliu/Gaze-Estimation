import torch
from torch import nn
from tqdm import tqdm

from config import device, num_workers
from data_gen import GazeEstimationDataset
from models import GazeEstimationModel
from utils import AverageMeter

if __name__ == '__main__':
    filename = 'gaze_estimation.pt'
    model = GazeEstimationModel()
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()  # train mode (dropout and batchnorm is used)

    val_loader = torch.utils.data.DataLoader(GazeEstimationDataset('val'), batch_size=1, shuffle=False,
                                             num_workers=num_workers)

    criterion = nn.SmoothL1Loss()

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
        loss2 = loss2
        loss = loss1 + loss2

        # Keep track of metrics
        losses.update(loss.item() * 1000, img.size(0))
        l_losses.update(loss1.item() * 1000, img.size(0))
        p_losses.update(loss2.item() * 1000, img.size(0))

    # Print status
    status = 'Validation\t' \
             'Loss {loss.avg:.5f}\t' \
             'Look Vec Loss {l_loss.avg:.5f}\t' \
             'Pupil Size Loss {p_loss.avg:.5f}\n'.format(loss=losses, l_loss=l_losses, p_loss=p_losses)
    print(status)
