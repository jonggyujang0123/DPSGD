"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from graphs.models.mnist import Mnist
from datasets.mnist import MnistDataLoader

from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True

import wandb
import os

os.environ['WANDB_SILENT']="false"

class MnistAgent(BaseAgent):

    def __init__(self, config, config_dict, gpu_ids, mode):
        wandb.init(project="Pytorch-Project-Template", entity="jonggyujang0123", config=config_dict, name=f'{config.agent}-lr-{config.learning_rate}', group=f'{config.agent}_{mode}')
        super().__init__(config)

#        if self.config.pretrained_encoder:
#            pretrained_enc = torch.nn.DataParallel(ERFNet(self.config.imagenet_nclasses)).cuda()
#            pretrained_enc.load_state_dict(torch.load(self.config.pretrained_model_path)['state_dict'])
#            pretrained_enc = next(pretrained_enc.children()).features.encoder
#        else:
#            pretrained_enc = None
        # define models
        self.model = Mnist()
#        self.model = ERF(self.config, pretrained_enc)
        self.mode = mode
        self.checkpoint_dir = config.checkpoint_dir + '/' + config.exp_name + '/' 
        if len(gpu_ids)>1:
            self.model = nn.DataParallel(self.model)


        # define data_loader
        self.data_loader = MnistDataLoader(config=config)

        # define loss
        self.loss = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.checkpoint_dir + self.config.checkpoint_file)
        # Summary Writer
        #self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='mnist')

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

    def run(self):
        """
        The main operator
        :return:
        """
        if self.mode == 'test':
            self.test()
        else:
            self.train()
#            self.test()


    def train(self):
        """
        Main training loop
        :return:
        """
        val_acc_current = 0 
        for self.epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            val_acc = self.validate()
            if val_acc > val_acc_current and self.config.save_best:
                self.save_checkpoint(self.config.checkpoint_file, is_best=True)
                val_acc_current = val_acc

            self.current_epoch += 1
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1
        wandb.log({"loss": loss.mean().detach().item()})
        wandb.watch(self.model)

    def test(self):
        """
        One cycle of model validation
        :return:
        """
        print("Loading checkpoints ... ")
        self.load_checkpoint(self.config.checkpoint_file)
        self.model.eval()
        test_loss = 0
        correct = 0
        example_images=[]
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_images.append(wandb.Image(data[0], caption="Pred: {} Truth: {}".format(pred[0].detach().item(), target[0])))
        wandb.log({"Examples":example_images})

        test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))
        wandb.log({"acc": 100. * correct / len(self.data_loader.test_loader.dataset)})

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.data_loader.val_loader.dataset)
        self.logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))
        return correct


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        if not self.mode == 'test':
            self.save_checkpoint(self.config.checkpoint_file)
        self.data_loader.finalize()
