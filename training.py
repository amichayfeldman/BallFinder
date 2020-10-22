import torch
import numpy as np
from Utils.help_funcs import write_to_csv
import os
import glob
import configparser
from Dataset.dataset import get_dataloaders
from Model.Model import BallDetector, init_weights
import torch.nn as nn
from Utils.Losses import FocalLoss, dice_loss
import cv2

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(3)
np.random.seed(2)


def save_model(model, epoch, output_path, best=False):
    if best:
        previous_best_pt = glob.glob(os.path.join(output_path, '*BEST.pt'))
        if len(previous_best_pt) > 0:
            os.remove(previous_best_pt[0])
        name = os.path.join(output_path, 'Model_stateDict__Epoch={}__BEST.pt'.format(epoch))
    else:
        name = os.path.join(output_path, 'Model_stateDict__Epoch={}.pt'.format(epoch))
    torch.save(model.state_dict(), os.path.join(output_path, name))


def train(model, train_dataloader, val_dataloader, epochs, criterion_loss, optimizer, scheduler, output_path,
          alpha, save_model=True, write_csv=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "Cuda is not working"

    train_loss_list, val_loss_list, lr_list, wd_list = [], [], [], []
    best_train_loss, best_val_loss = np.inf, np.inf
    if not os.path.isdir(os.path.join(output_path, 'saved_checkpoints')):
        os.makedirs(os.path.join(output_path, 'saved_checkpoints'))

    model.train()
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        # --- TRAIN:  --- #
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'].to(device), data['gt'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.type(torch.FloatTensor).to(device))
            sampled_labels = torch.nn.functional.interpolate(labels.type(torch.FloatTensor).unsqueeze(1),
                                                             scale_factor=(outputs.shape[2] / labels.shape[1],
                                                             outputs.shape[3] / labels.shape[2]), mode='bicubic')
            sampled_labels[sampled_labels != 0] = 1
            ce_loss = criterion_loss(outputs.cuda(), sampled_labels.squeeze().cuda().long())
            # dice_score = dice_loss(output=outputs.cuda(), target=sampled_labels.squeeze().cuda().long())
            # loss = alpha * ce_loss + (1 - alpha) * torch.autograd.Variable(dice_score, requires_grad=True)
            ce_loss.backward()
            optimizer.step()
            running_loss += ce_loss.item()
        train_loss = running_loss / (i + 1)
        train_loss_list.append(train_loss)
        ##################

        model.eval()
        # --- VAL:  --- #
        with torch.no_grad():
            val_running_loss = 0.0
            for val_i, val_data in enumerate(val_dataloader):
                inputs, labels = val_data['image'].to(device), val_data['gt'].to(device)
                outputs = model(inputs.type(torch.FloatTensor).to(device))
                sampled_labels = torch.nn.functional.interpolate(labels.type(torch.FloatTensor).unsqueeze(1),
                                                                 scale_factor=(outputs.shape[2] / labels.shape[1],
                                                                               outputs.shape[3] / labels.shape[2]),
                                                                 mode='bicubic')
                sampled_labels[sampled_labels != 0] = 1
                ce_loss = criterion_loss(outputs.cuda(), sampled_labels.squeeze().cuda().long())
                # dice_score = dice_loss(output=outputs.cuda(), target=sampled_labels.squeeze().cuda().long())
                # loss = alpha * ce_loss + (1 - alpha) * dice_score
                val_running_loss += ce_loss.item()
            val_loss = val_running_loss / (val_i + 1)
            val_loss_list.append(val_loss)
        ##################

        # --- Save results to csv ---#
        # current_lr = optimizer.param_groups[-1]['lr']
        # current_wd = optimizer.param_groups[-1]['weight_decay']
        # epoch_results = [epoch, train_loss, val_loss, current_lr, current_wd]
        # write_to_csv(os.path.join(output_path, 'results.csv'), epoch_results)
        lr_list.append(optimizer.param_groups[-1]['lr'])
        wd_list.append(optimizer.param_groups[-1]['weight_decay'])
        ##############################

        # --- Save model checkpoint ---#
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                save_model(model=model, epoch=epoch, output_path=os.path.join(output_path, 'saved_checkpoints'), best=True)
        elif epoch % 10 == 0:
            if save_model:
                save_model(model=model, epoch=epoch, output_path=os.path.join(output_path, 'saved_checkpoints'), best=False)
        ###############################
        if scheduler is not None:
            scheduler.step(val_loss)
        train_dataloader.dataset.change_img_size()
        val_dataloader.dataset.change_img_size()

        #  PRINT:  #
        print("Epoch {}:  train loss: {:.5f}, val loss: {:.5f}".format(epoch, train_loss, val_loss))
    if write_csv:
        write_to_csv(os.path.join(output_path, 'results.csv'), [list(range(epochs)), train_loss_list, val_loss_list,
                                                            lr_list, wd_list])
    return train_loss_list, val_loss_list  # return last train loss and the best val loss


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    batch_size = config.getint('Params', 'batch_size')
    lr = config.getfloat('Params', 'lr')
    wd = config.getfloat('Params', 'wd')
    alpha = wd = config.getfloat('Params', 'alpha')
    epochs = config.getint('Params', 'epochs')
    output_folder = config['Paths']['output_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # --- Dataset --- #
    train_images = glob.glob(os.path.join(config['input_images']['train'], '*/*.jpg'))
    val_images = glob.glob(os.path.join(config['input_images']['val'], '*/*.jpg'))
    train_gt = glob.glob(os.path.join(config['ground_truth']['train'], '*/*/*.png'))
    val_gt = glob.glob(os.path.join(config['ground_truth']['val'], '*/*/*.png'))

    datasets_imgs_folders = {'train': train_images, 'val': val_images}
    gt_imgs_folders = {'train': train_gt, 'val': val_gt}

    train_dataloader, val_dataloader, _ = get_dataloaders(dataset_dict=datasets_imgs_folders,
                                                          gt_dict=gt_imgs_folders,
                                                          batch_size=batch_size, num_workers=0, config=config)
    #####################

    # --- Model --- #
    checkpoint = config['Paths']['model_checkpoint']
    if len(checkpoint) > 0:
        model = BallDetector(config=config)
        model.load_state_dict(torch.load(checkpoint))
    else:
        model = BallDetector(config=config)
        model.apply(init_weights)
    #####################

    # --- Loss function --- #
    loss_bce = nn.BCELoss()
    focal_loss = FocalLoss(gamma=2, alpha=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    #########################

    # --- send training --- #
    train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=epochs,
          criterion_loss=focal_loss, optimizer=optimizer, scheduler=scheduler, output_path=output_folder,
          alpha=alpha, save_model=True)


if __name__ == '__main__':
    main()