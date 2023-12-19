
import argparse
import os
import glob
import logging

import numpy as np
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets import DatasetFolder
# from torchvision import transforms
from sklearn import metrics

try:
    from tensorboardX import SummaryWriter
except ImportError:
    pass

import unet

HAS_CUDA = torch.cuda.is_available()

logger = logging.getLogger(__name__)


class SpectralisDataset(Dataset):
    
    def __init__(self, base_path):
        super().__init__()
        
        image_glob = os.path.join(base_path, "**/image_*.tif")
        self.image_files = sorted(glob.glob(image_glob))
        
        label_glob = os.path.join(base_path, "**/label_*.tif")
        self.label_files = sorted(glob.glob(label_glob))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        image = imageio.imread(self.image_files[idx])
        label = imageio.imread(self.label_files[idx])
        
        # Add extra dimension to account for the single channel
        image = image[None]
        
        return image, label


def _training_transforms(image, label):
    
    # Random horizontal flip
    if np.random.uniform(0, 1) > 0.5:
        image = image[..., ::-1]
        label = label[..., ::-1]
    
    image = np.ascontiguousarray(image, dtype=np.float32)
    label = np.ascontiguousarray(label, dtype=np.int64)
    
    return image, label


def _testing_transform(image, label):
    
    image = np.ascontiguousarray(image, dtype=np.float32)
    label = np.ascontiguousarray(label, dtype=np.int64)
    
    return image, label


def build_training_dataset():
    
    dataset = SpectralisDataset("/home/pablo.marquez/data/retouch/spectralis_bscans/train")
    
    # Compute class weights
    bin_counter = unet.BinCounter()
    for _, label in dataset:
        bin_counter.update(label)
    class_weights = unet.invfreq_weights(bin_counter)
    
    # Transformed dataset (with random-cropping, horizontal flipping, rotation...)
    dataset = unet.TransformedDataset(dataset, _training_transforms)
    
    return dataset, class_weights


def build_testing_dataset():
    dataset = SpectralisDataset("/home/pablo.marquez/data/retouch/spectralis_bscans/test")
    dataset = unet.TransformedDataset(dataset, _testing_transform)
    return dataset


def train_step(sampler,
               class_weights: torch.Tensor,
               network: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    
    # Put the network in train mode
    network.train()
    
    x, y = sampler()
    
    # Move minibatch to the gpu
    x = x.to(device)
    y = y.to(device)
    
    pred = network(x)
    loss = torch.nn.functional.nll_loss(pred, y, class_weights)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {"loss": loss.item()}


# Disable gradient computation in the validation function
@torch.no_grad()
def evaluate(test_loader: DataLoader,
             class_weights: torch.Tensor,
             network: torch.nn.Module,
             device: torch.device):
    
    # Put the network in evaluation mode
    network.eval()
    
    xs = []
    y_true = []
    y_pred = []
    losses = []
    
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        pred = network(x)
        loss = torch.nn.functional.nll_loss(pred, y, weight=class_weights, reduction='none')
        
        losses.append(loss.mean(dim=(1, 2)))
        xs.append(x.cpu().numpy())
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    
    losses = torch.cat(losses, dim=0)
    loss = losses.mean()
    
    xs = np.concatenate(xs, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    num_classes = y_pred.shape[1]
    aps = []
    for c in range(1, num_classes):
        ravel_y_true = (y_true == c).ravel()
        ravel_y_pred = y_pred[:, c].ravel()
        ap = metrics.average_precision_score(ravel_y_true, ravel_y_pred)
        
        aps.append(ap)
    
    aps = np.array(aps)
    mAP = np.mean(aps)
    
    logger.info("Per-class AP: {}".format(aps))
    logger.info("mAP: {}".format(mAP))
    
    # Pick a random scan for plotting
    idx = np.random.randint(0, len(y_true))
    example_x = xs[idx, 0]
    example_y = y_true[idx]
    example_preds = np.exp(y_pred[idx])
    
    return {
        "loss": loss.item(),
        "map": mAP,
        **{"AP_{}".format(i): v for i, v in enumerate(aps, 1)},
        "x": example_x,
        "y": example_y,
        **{"pred_{}".format(i): v for i, v in enumerate(example_preds)}
    }


def main(
        output_path="saved_bn{batch_norm}_sc{scaled_conv}_opt{optimizer}",
        batch_size=2,
        multigpu=False,
        batch_norm=False,
        scaled_conv=False,
        optimizer='Adam',
        continue_from=-1
    ):
    
    output_path = output_path.format(**locals())
    
    unet.config_logger(os.path.join(output_path, "main.log"))
    
    device = torch.device("cuda") if HAS_CUDA else torch.device("cpu")
    
    logger.info("Loading data...")
    train_dataset, class_weights = build_training_dataset()
    test_dataset = build_testing_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    num_classes = len(class_weights)
    
    # Create the network
    logger.info("Creating U-Net...")
    unet_config = unet.UNetConfig(
        steps=4,
        ndims=2,
        num_classes=num_classes,
        first_layer_channels=64,
        num_input_channels=1,
        two_sublayers=True,
        border_mode='same',
        batch_normalization=batch_norm,
        scaled_convolution=scaled_conv
    )
    
    unet_clsf = unet.UNetClassifier(unet_config)
    unet_clsf.to(device)
    
    if multigpu:
        # Run on multiple GPUs
        logger.info("Using DataParallel: {} GPUs available".format(torch.cuda.device_count()))
        unet_clsf = torch.nn.DataParallel(unet_clsf)
    
    # Build optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(unet_clsf.parameters(), lr=1e-4)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(unet_clsf.parameters(), lr=1e-4)
    
    # Move class_weights to the gpu
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    # sampler is a function that returns a minibatch from train_loader every time is called
    sampler = unet.build_sampler(train_loader)
    
    # TensorboardX writer
    summary_writer = SummaryWriter(os.path.join(output_path, "summary"), purge_step=continue_from + 1)
    
    training_manager = unet.TrainingManager(
        training_step=lambda _: train_step(sampler, class_weights, unet_clsf, optimizer, device),
        save_every=500,
        save_path=output_path,
        managed_objects=unet.managed_objects({
            "network": unet_clsf,
            "optimizer": optimizer
        }),
        test_function=lambda _: evaluate(test_loader, class_weights, unet_clsf, device),
        test_every=500,
        summary_writer=summary_writer
    )
    
    if continue_from >= 0:
        logger.warning("Continuing training from iteration {}".format(continue_from))
        training_manager.load(continue_from)
    
    training_manager.train(num_iters=10000, print_every=50)
    
    # import IPython
    # IPython.embed()
    
    # For training, run
    #   training_manager.train(num_iters=10, print_every=5)
    
    # You can save managed objects (in this case the network
    # and the optimizer) at the current iteration with
    #   training_manager.save()
    # You can load managed objects as they were at a given iteration with
    #   training_manager.load(50000)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--multigpu", action='store_true')
    parser.add_argument("--batch-norm", action='store_true')
    parser.add_argument("--scaled-conv", action='store_true')
    parser.add_argument("--continue-from", type=int, default=-1)
    
    args = parser.parse_args()
    
    np.set_printoptions(precision=4)
    main(**vars(args))
