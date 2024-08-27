import argparse
import math
import os
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from src.metrics import evaluate_metrics
import glob
from src.lstm import TransformerModel


from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    read_tiff,
    load_norm,
    check_folder,
    plot_figures,
    oversamp,
    add_padding
)

from src.multicropdataset import DataLoaderTabular
#import segmentation_models_pytorch as smp

logger = getLogger('swav_model')

parser = argparse.ArgumentParser(description="Training of Transformer")

data_dir='byYear_v2'  #    byfieldID,  byYear 
y = 'out2020'  #    byfieldID,  out2022  
inputs = 'RS'
fold = '_fold4'
results = 'years' #  byfieldID  years  ,byspace
predname = f'{y}_{inputs}_Transf_v2'

#########################
#### data parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=f"./exp_{results}/{y}_{inputs}_Transf_v2",
                    help="experiment dump path for checkpoints and log")

parser.add_argument('--data_train', type=str, default=f'./data/{data_dir}/{y}_{inputs}_data_train2.npy',
                        help="Path containing the raster image")
parser.add_argument('--target_train',type=str, default=f'./data/{data_dir}/{y}_{inputs}_target_train2.npy', 
                    help="Path containing the refrence label image for training")
parser.add_argument('--data_val', type=str, default=f'./data/{data_dir}/{y}_{inputs}_data_val2.npy',
                        help="Path containing the raster image")
parser.add_argument('--target_val',type=str, default=f'./data/{data_dir}/{y}_{inputs}_target_val2.npy', 
                    help="Path containing the refrence label image for training")
parser.add_argument('--data_test', type=str, default=f'./data/{data_dir}/{y}_{inputs}_data_test.npy',
                        help="Path containing the raster image")
parser.add_argument('--target_test',type=str, default=f'./data/{data_dir}/{y}_{inputs}_target_test.npy', 
                    help="Path containing the refrence label image for training")

##############################
parser.add_argument("--size_crops", type=int, default=128,  
                    help="Size of the input tile for the network")
parser.add_argument('--image_test', type=str, default='./data/Test_stack_npy',
                        help="Path containing the raster image")
parser.add_argument('--ref_test',type=str, default='./data/Target_Test_npy', 
                    help="Path containing the refrence label image for training")

#########################
#### model parameters  ###
#########################)
parser.add_argument("--hidden_size", default=128, type=int, 
                    help="Number of LSTM units [32, 64, 128, 256, 512]")
parser.add_argument("--num_layers", default=1, type=int, 
                    help="Number of LSTM layer [1,2,3]")
parser.add_argument("--opt", default='Adam', type=str, 
                    help="Optimizer, SGD or Adam")
parser.add_argument("--sigm_out", default=True, type=bool, 
                    help="Set to True to apply Sigmoid")

parser.add_argument("--output_size", default=1, type=int, 
                    help="Number of output predictions")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run") 
parser.add_argument("--batch_size", default=256, type=int, ## batch size:16,32,64,128,256,512,1024
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.01, type=float, help="base learning rate") ####################### IMPORTANTE
parser.add_argument("--final_lr", type=float, default=0.0000001, help="final learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters  ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

##########################
#### others parameters  ###
##########################

parser.add_argument("--workers", default=0, type=int,
                    help="number of data loading workers")
parser.add_argument("--seed", type=int, default=31, help="seeds")


def main(hs,bl,nl,op,so):
    global args, figures_path
    args = parser.parse_args()
    
    args.hidden_size = hs
    args.base_lr = bl
    args.num_layers = nl
    args.opt = op
    args.sigm_out = so
    
    args.dump_path = os.path.join(args.dump_path,'{}_{}_{}_{}_{}'.format(hs,bl,nl,op,so))
    
    check_folder(args.dump_path)
    fix_random_seeds(args.seed)
    
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    
    figures_path = os.path.join(args.dump_path, 'prediction')
    check_folder(figures_path)
    
    
    ######### Define Loader for training, validation and testing ############
    data_train = np.load(args.data_train)
    data_train = np.moveaxis(data_train, 2,1)
    target_train = np.load(args.target_train)
        
    data_val = np.load(args.data_val)
    data_val = np.moveaxis(data_val, 2,1)
    target_val = np.load(args.target_val)
    
    data_test = np.load(args.data_test)
    data_test = np.moveaxis(data_test, 2,1)
    target_test = np.load(args.target_test)
    
    # build data for training
    train_dataset = DataLoaderTabular(
        data_train,
        target_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    
    
    # build data for validation
    val_dataset = DataLoaderTabular(
        data_val,
        target_val
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )
    

    # build data for testing
    test_dataset = DataLoaderTabular(
        data_test,
        target_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    
     
    logger.info("Building data done with {} images loaded.".format(len(train_loader)))
    
    # build model
    model = TransformerModel(c_in = data_train.shape[1], 
                             c_out = args.output_size, 
                             d_model=args.hidden_size, 
                             n_head=1, 
                             d_ffn=128, 
                             dropout=0.0, 
                             activation="relu", 
                             n_layers=args.num_layers,
                             sigm_out = args.sigm_out)
        
        
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total trainable parameters {} ".format(pytorch_total_params))


    # copy model to GPU
    model = model.cuda()
    
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.wd
        )
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.wd
        )
    
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")


    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]


    cudnn.benchmark = True

    best_val = 1000000000.0
    cont_early = 0
    patience = 100

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train the network
        scores_tr = train(train_loader, model, optimizer, epoch, lr_schedule)
        
        logger.info("============ Predict test at epoch %i ... ============" % epoch)

        if epoch < args.epochs-1:
            
            _ = predict(train_loader, model, epoch, name="Train")
            scores_val = predict(val_loader, model, epoch, name="Validation")
            _ = predict(test_loader, model, epoch, name="Test")
        
        training_stats.update(scores_tr)
        
        is_best = scores_val <= best_val
            
        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if is_best:
                logger.info("============ Saving best models at epoch %i ... ============" % epoch)
                cont_early = 0
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                )
                best_val = scores_val
            else:
                cont_early+=1
                
                
            if cont_early == patience:
                logger.info("============ Early Stop at epoch %i ... ============" % epoch)
                break
            
    
    _ = predict(train_loader, model, epoch=args.epochs, last_epoch=True, name="Train")
    _ = predict(val_loader, model, epoch=args.epochs, last_epoch=True, name="Validation")
    _ = predict(test_loader,model,  epoch=args.epochs, last_epoch=True, name="Test")
            


def train(train_loader, model, optimizer, epoch, lr_schedule):

    model.train()
    loss_avg = AverageMeter()
    

    # define losses
    # criterion = nn.L1Loss().cuda()
    criterion = nn.MSELoss().cuda()
    

    for it, (inp_img, ref) in enumerate(train_loader):      

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ forward pass and loss ... ============
        # compute model loss and output
        inp_img = inp_img.cuda(non_blocking=True)
        ref = ref.cuda(non_blocking=True)
        
        
        # calculate losses
        out_batch = model(inp_img)
        loss = criterion(out_batch[:,0], ref)
        

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()
        
        # performs updates using calculated gradients
        optimizer.step()
        
        # update the average loss
        loss_avg.update(loss.item())

        # Evaluate summaries only once in a while
        if it % 50 == 0:
            
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    loss=loss_avg,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            

            
    return (epoch, loss_avg.avg)

def predict(test_loader, model, epoch, last_epoch=False, name=None):

    model.eval()
    loss_avg = AverageMeter()
    rmse_avg = AverageMeter()

    # define losses

    # criterion = nn.L1Loss().cuda()
    criterion = nn.MSELoss().cuda()     
    final_pred = []
    for it, (inp_img, ref) in enumerate(test_loader):      
 
        # ============ forward pass and loss ... ============
        # compute model loss and output

        inp_img = inp_img.cuda(non_blocking=True)
        ref = ref.cuda(non_blocking=True)

        # calculate losses
        out_batch = model(inp_img)
        loss = torch.sqrt(criterion(out_batch[:,0]*7000, ref*7000))

        if last_epoch:
            final_pred.extend((out_batch[:,0]*7000).data.cpu().numpy().tolist())

        mean = (ref*7000).mean()
        rrmse = (loss/mean)*100

        # update the average loss

        loss_avg.update(loss.item())
        rmse_avg.update(rrmse.item())
             
    logger.info(
        "Epoch: [{0}]\t\t"
        "{setname} RRMSE {rmse.val:.4f} ({rmse.avg:.4f})".format(
            epoch,
            setname = name,
            rmse = rmse_avg,
        )
    )

    if last_epoch:
        check_folder(os.path.join(figures_path,predname))
        np.save(os.path.join(figures_path,predname,name), np.array(final_pred))
        
    return loss_avg.avg



if __name__ == "__main__":
    hidden_size = [64]
    base_lr = [ 0.0001]
    num_layers = [1]
    opt = ['Adam']
    sigm_out = [True]
    for nl in num_layers:
      for hs in hidden_size:
          for bl in base_lr:
                  for op in opt:
                      for so in sigm_out:
                          main(hs,bl,nl,op,so)