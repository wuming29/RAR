import os
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging as log
import numpy
import tqdm
import pickle
from utils import batch_data_to_device

def train(model, loaders, args):
    maxacc = 0
    log.info("training...")
    BCELoss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_sigmoid = torch.nn.Sigmoid()
    train_len = len(loaders['train'].dataset)
    for epoch in range(args.n_epochs):
        loss_all = 0
        acc_all = 0
        auc_all = 0
        count_all = 0
        for step, data in enumerate(loaders['train']):
            with torch.no_grad():
                x, y = batch_data_to_device(data, args.device)
            model.train()
            hat_y_prob = None
            full_pre = []
            logits, full_pre = model(x)
            hat_y_prob = train_sigmoid(logits)
            '''calculation cross entropy'''
            
            loss = BCELoss(logits, y)

            optimizer.zero_grad() 
            loss.backward()

            optimizer.step()
            # step += 1
            
            with torch.no_grad():
                loss_all += loss.item()
                hat_y_bin = (hat_y_prob > 0.5).int()
                acc = accuracy_score(y.int().cpu().numpy(), hat_y_bin.cpu().numpy())
                fpr, tpr, thresholds = metrics.roc_curve(y.detach().int().cpu().numpy(), hat_y_prob.detach().cpu().numpy(),pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auc_all += auc
                acc_all += acc
                count_all += 1

                # if step % args.eval_every == 1:
                #     show_loss = loss_all / train_len
                #     show_acc = acc_all / count_all
                #     show_auc = auc_all / count_all
                #     acc, auc, auroc = evaluate(model, loaders['valid'], args.device)
                #     log.info('Epoch: {:03d}, Step: {:03d}, Loss: {:.7f}, train_acc: {:.7f}, train_auc: {:.7f}, acc: {:.7f}, auc: {:.7f}'.format(epoch, step, show_loss, show_acc, show_auc, acc, auc))

        show_loss = loss_all / train_len
        show_acc = acc_all / count_all
        show_auc = auc_all / count_all
        acc, auc = evaluate(model, loaders['valid'], args.device)
        log.info('Epoch: {:03d}, Loss: {:.7f}, train_acc: {:.7f}, train_auc: {:.7f}, acc: {:.7f}, auc: {:.7f}'.format(epoch, show_loss, show_acc, show_auc, acc, auc))

        if acc > maxacc:
            torch.save(model, 'dkt_assist09_simulator.pt')
            log.info('model updated, acc: {:.7f}, maxacc: {:.7f}'.format(acc, maxacc))
            maxacc = acc


        # if args.save_every > 0 and epoch % args.save_every == 0:
        #     torch.save(model, os.path.join(args.run_dir, 'params_%i.pt' % epoch))


def evaluate(model, loader, device):
    model.eval()
    rre_list = []
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, hat_y_list = [], []
    with torch.no_grad():
        for data in loader:
            x, y = batch_data_to_device(data, device)
       
            hat_y_prob, full_pre = model(x)
            y_list.append(y)
            hat_y_list.append(eval_sigmoid(hat_y_prob))

    y_tensor = torch.cat(y_list, dim = 0).int()
    hat_y_prob_tensor = torch.cat(hat_y_list, dim = 0)
    acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.5).int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    return acc, auc


