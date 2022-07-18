import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.utils import save_image
plt.style.use('ggplot')

def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20*math.log10(max_val/rmse)
        return PSNR


def save_plot(train_loss, val_loss, train_psnr, val_psnr):
    #loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.png')
    #plt.show()
    # psnr plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('../outputs/psnr.png')
    #plt.show()
    plt.close()


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    torch.save({
                'epoch': epochs+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"../outputs/model_ckpt.pth")


def save_model_state(model):
    # save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), '../outputs/model.pth')


def save_validation_results(outputs, epoch, batch_iter):
    save_image(
        outputs,
        f"../outputs/valid_results/val_sr_{epoch}_{batch_iter}.png"
    )
