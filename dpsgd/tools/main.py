"""========Default import===="""
#from __future__ import absolute_import, division, print_function
#import logging
import argparse
import os
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
import shutil
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
""" ==========END================"""

""" =========Configurable ======="""
from models.resnet50 import R_50_MNIST as resnet
#os.environ["WANDB_SILENT"] = 'true'

""" ===========END=========== """

#logger = logging.getLogger("dpsgd")

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
            type=str, 
            help="Configuration file in configs.")
    parser.add_argument("--multigpu",
            type=bool, 
            help="Local rank. Necessary for using the torch.distributed.launch utility.",
            default= False)
    parser.add_argument("--resume",
            default= 0,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--test", 
            type=bool,
            default = False,
            help="if test, choose True.")
    args = parser.parse_args()
    return args

def get_data_loader(cfg, args):
    if cfg.dataset in ['cifar100', 'cifar10']:
        from datasets.cifar import get_loader_cifar as get_loader

    return get_loader(cfg, args)





def load_ckpt(checkpoint_fpath, map_location,is_best =False):
    """
    Latest checkpoint loader
        checkpoint_fpath : 
    :return: dict
        checkpoint{
            model,
            optimizer,
            epoch,
            scheduler}
    example :
    """
    if is_best:
        ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    else:
        ckpt_path = checkpoint_fpath+'/'+'best.pt'
    try:
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(checkpoint_fpath, map_location=map_location)
    except:
        print(f"No checkpoint exists from '{ckpt_path}'. Skipping...")
        print("**First time to train**")
    return checkpoint


def save_ckpt(checkpoint_fpath, checkpoint, is_best=False):
    """
    Checkpoint saver
    :checkpoint_fpath : directory of the saved file
    :checkpoint : checkpoiint directory
    :return:
    """
    ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    # Save the state
    if not os.path.exists(checkpoint_fpath):
        os.makedirs(checkpoint_fpath)
    torch.save(checkpoint, ckpt_path)
    # If it is the best copy it to another file 'model_best.pth.tar'
#    print("Checkpoint saved successfully to '{}' at (epoch {})\n"
#        .format(ckpt_path, checkpoint['epoch']))
    if is_best:
        ckpt_path_best = checkpoint_fpath+'/'+'best.pt'
        print("This is the best model\n")
        shutil.copyfile(ckpt_path,
                        ckpt_path_best)




def train(wandb, args, cfg, ddp_model):
    ## Setting 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    train_loader, test_loader = get_data_loader(cfg=cfg, args=args)
    if cfg.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    val_acc = .0; best_acc = .0; start_epoch =0
    AvgLoss = AverageMeter()
    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume == True:
        if args.multigpu:
            map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
        else:
            map_location = {"cuda:0": "cuda:0"}
        ckpt = load_ckpt(cfg.ckpt_fpath, map_location)
        ddp_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']

        

    # Prepare dataset and dataloader

    for epoch in range(start_epoch, cfg.epochs):
        if args.local_rank not in [-1,1]:
            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        ddp_model.train()
        tepoch = tqdm(train_loader, 
                      disable=args.local_rank not in [-1,0])
        for batch, data in enumerate(tepoch):
            optimizer.zero_grad()
            inputs, labels = data[0].to(cfg.device), data[1].to(cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                outputs = ddp_model(inputs)
                loss =criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            #optimizer.step()
            AvgLoss.update(loss.item(), n =len(data))
            if args.local_rank in [-1,0]:
                tepoch.set_description(f'Epoch {epoch}: train_loss: {AvgLoss.avg:2.2f}, lr : {scheduler.get_lr()[0]:.2E}')

        # save and evaluate model routinely.
        if epoch % cfg.interval_val == 0:
            if args.local_rank in [-1, 0]:
                val_acc = evaluate(model=ddp_model, device=cfg.device, val_loader=test_loader, use_amp = cfg.use_amp)
                ckpt = {
                        'model': ddp_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch
                        }
                save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt)
                if best_acc < val_acc.avg:
                    save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt, is_best=True)
                    best_acc = val_acc.avg
                if cfg.wandb.active:
                    wandb.log({
                        "loss": AvgLoss.val,
                        "val_acc" : val_acc.avg
                        })
                print("-"*75+ "\n")
                print(f"| {epoch}-th epoch, training loss is {AvgLoss.avg}, and Val Accuracy is {val_acc.avg} = {val_acc.sum}/{val_acc.count}\n")
                print("-"*75+ "\n")
        AvgLoss.reset()
        tepoch.close()


def evaluate(model, val_loader, device, use_amp):
    model.eval()
    acc = AverageMeter()
    valepoch = tqdm(val_loader, 
                  unit= " batch")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valepoch):
            inputs = inputs.to(device, non_blocking=True)
            # compute the output
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(inputs).detach().cpu()

            # Measure accuracy
            acc_batch = get_accuracy(output=output, label=labels)
            acc.update(acc_batch, n=inputs.size(0))
            valepoch.set_description(f'Validation Acc: {acc.avg:2.2f}')
    valepoch.close()
    return acc

def test(args, cfg, model):
    _, test_loader = get_data_loader(cfg=cfg, args= args)
    print("Loading checkpoints ... ")
    map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
    ckpt = load_ckpt(checkpoint_fpath= cfg.fpath, map_location = map_location, is_best = True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    acc = AverageMeter()
    #example_images = []
    with torch.no_grad():
        if args.local_rank in [-1,0]:
            tepoch = tqdm(test_loader, unit= " batch")
            tepoch.set_description("TEST")
        else:
            tepoch = tqdm(test_loader, unit= " batch", disable=True)
        for batch, (inputs, labels) in enumerate(tepoch):
            inputs = inputs.to(cfg.device, non_blocking=True)
            # compute the output
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                output = model(inputs).detach().cpu()

            # Measure accuracy
            acc_batch = get_accuracy(output=output, label=labels)
            acc.update(acc_batch, n=inputs.size(0))
            #example_images.append(wandb.Image(data[0], caption="Pred: {} Truth: {}".format(pred[0].detach().item(), target[0])))

        #wandb.log({"Examples":example_images})
        print("-"*75+ "\n")
        print(f"| Testset accuracy is {acc.avg} = {acc.sum}/{acc.count}\n")
        print("-"*75+ "\n")

    return acc


def main():
    args = parse_args()
    cfg = edict(yaml.safe_load(open(args.config)))
    if args.multigpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = -1

    if args.local_rank in [-1,0] and args.test==False and cfg.wandb.active:
        wandb.init(project = cfg.wandb.project,
                   entity = cfg.wandb.id,
                   config = dict(cfg),
                   name = f'{cfg.wandb.name}_lr:{cfg.lr}_fp16:{cfg.use_amp}',
                   group = cfg.dataset
                   )
    if args.local_rank in [-1,0]:
        print(cfg)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = cfg.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    if args.multigpu: 
        torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend="gloo")
        cfg.device = torch.device("cuda:{}".format(args.local_rank))
    else:
        cfg.device = torch.device("cuda:0")

    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = resnet(num_classes=cfg.num_classes)
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model = model.to(cfg.device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter is {pytorch_total_params:.2E}')
    if args.multigpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    if args.test:
        test(args, cfg, model)
    else:
        train(wandb, args, cfg, model)
    


if __name__ == '__main__':
    main()
