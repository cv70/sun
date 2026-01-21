import torch
from utils.loss import calc_loss_loader, calc_sft_loss_loader, calc_dpo_loss_loader

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def evaluate_sft_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_sft_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_sft_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def evaluate_dpo_model(model, ref_model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_dpo_loss_loader(train_loader, model, ref_model, device, num_batches=eval_iter)
        val_loss = calc_dpo_loss_loader(val_loader, model, ref_model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
