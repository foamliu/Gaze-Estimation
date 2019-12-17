import torch
from torch import nn
from tqdm import tqdm

from config import device, batch_size, num_workers
from data_gen import GazeEstimationDataset
from models import GazeEstimationModel
from utils import AverageMeter

if __name__ == '__main__':
    filename = 'gaze_estimation.pt'
    model = GazeEstimationModel()
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()  # train mode (dropout and batchnorm is used)

    val_loader = torch.utils.data.DataLoader(GazeEstimationDataset('val'), batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    criterion = nn.SmoothL1Loss()
    losses = AverageMeter()

    # Batches
    for (img, label) in tqdm(val_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.float().to(device)  # [N, 3]

        # Forward prop.
        with torch.no_grad():
            output = model(img)  # embedding => [N, 3]

        # Calculate loss
        loss = criterion(output, label)

        # Keep track of metrics
        losses.update(loss.item() * 1000, img.size(0))

    # Print status
    status = 'Validation\t Loss {loss.avg:.5f}\n'.format(loss=losses)
    print(status)
