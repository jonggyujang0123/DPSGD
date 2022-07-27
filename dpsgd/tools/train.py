"""========Default import===="""
#from __future__ import absolute_import, division, print_function
import logging
import argparse
import os
import time 
from utils import set_random_seeds, get_accuracy, AverageMeter
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
""" ==========END================"""

""" =========Configurable ======="""

from configs.configs_base import BaseConfig
from models.mnist import R_50_MNIST as resnet
from datasets.cifar import get_loader
os.environ["WANDB_SILENT"] = 'true'

""" ===========END=========== """




def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank",type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=0)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    args = parser.parse_args()
    return args




def load_checkpoint(self, file_name="checkpoint.pth.tar"):
    """
    Latest checkpoint loader
    :param file_name: name of the checkpoint file
    :return:
    example :
    """
    filename = self.checkpoint_dir + file_name
    try:
        self.logger.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)

        self.current_epoch = checkpoint['epoch']
        self.current_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
              .format(self.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
    except OSError as e:
        self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.checkpoint_dir))
        self.logger.info("**First time to train**")

def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
    """
    Checkpoint saver
    :param file_name: name of the checkpoint file
    :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
    :return:
    example
    """
    state = {
        'epoch': self.current_epoch + 1,
        'iteration': self.current_iteration,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
    }
    # Save the state
    if not os.path.exists(self.checkpoint_dir):
        os.makedirs(self.checkpoint_dir)
    torch.save(state, self.checkpoint_dir + file_name)
    # If it is the best copy it to another file 'model_best.pth.tar'
    self.logger.info("Checkpoint saved successfully to '{}' at (epoch {})\n"
        .format(self.checkpoint_dir, self.epoch))
    if is_best:
        self.logger.info("This is the best model\n")
        shutil.copyfile(self.checkpoint_dir + file_name,
                        self.checkpoint_dir + 'model_best.pth.tar')




def train(args, cfg, ddp_model):
    device = torch.device("cuda:{}".format(args.local_rank))
    train_loader, test_loader = get_loader(cfg=cfg)
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume == True:
        map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
        ddp_model.load_state_dict(torch.load(cfg.model_filepath, map_location=map_location))

    # Prepare dataset and dataloader

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay = cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_map)
    if cfg.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs)


    best_acc = 0
    for epoch in range(cfg.epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        # save and evaluate model routinely.
        if epoch % cfg.interval_val == 0:
            if cfg.local_rank == 0:
                ddp_model.evaluate()
                val_acc = evaluate(model=ddp_model, device=device, val_loader=test_loader, cfg.use_map)
                torch.save(ddp_model.state_dict(), cfg.model_filepath)
                if best_acc < val_acc:
                    torch.save(ddp_model.state_dict(), cfg.best_model_filepath)
                    best_acc = val_acc
        ddp_model.train()

        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.use_map):
                outputs = ddp_model(inputs)
                loss =criterion(outputs, labels)
            scaler(loss.backward())
            scaler.step()
            scaler.update()
            optimizer.zero_grad()
            #optimizer.step()
        
            if args.local_rank in [-1,0]:

                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
        wandb.log({"loss": loss.mean().detach().item()})


def evaluate(model, val_loader, device):

    model.eval()
    acc = AverageMeter()
    with torch.no_grad():
        start = time.time()
        for i, (inputs, labels) in enumerate(val_loader):
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.to(device, non_blocking=True)
            # compute the output
            output = model(inputs)

            # Measure accuracy
            acc_batch = get_accuracy(output=output, label=labels)
            acc.update(acc_batch, n=inputs.size(0))
    return acc

def test(model, test_loader, device):
    print("Loading checkpoints ... ")
    self.load_checkpoint(self.config.checkpoint_file)
    model.eval()
    acc = AverageMeter()
    with torch.no_grad():
        start = time.time()
        example_images=[]
        for i, (inputs, labels) in enumerate(val_loader):
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.to(device, non_blocking=True)
            # compute the output
            output = model(inputs)
            example_images.append(wandb.Image(data[0], caption="Pred: {} Truth: {}".format(pred[0].detach().item(), target[0])))

            # Measure accuracy
            acc_batch = get_accuracy(output=output, label=labels)
            acc.update(acc_batch, n=inputs.size(0))
        wandb.log({"Examples":example_images})
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))
        wandb.log({"acc": 100. * correct / len(self.data_loader.test_loader.dataset)})
    return acc


def main():
    logger = logging.getLogger()
    cfg = BaseConfig()
    args = parse_args()
    """ Training the model """
    if args.local_rank in [-1,0]:
        wandb.init(project = cfg.project,
                   entity = cfg.wandb_id,
                   config = cfg.to_dict(),
                   name = cfg.name,
                   group = cfg.group,
                   )
    # set cuda flag
    if not torch.cuda.is_available():
        logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = cfg.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = resnet()
    # model.load_from(np.load(<path>))
    device = torch.device("cuda:{}".format(args.local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    train(args, cfg, model)
    


if __name__ == '__main__':
    main()
