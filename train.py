import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from torch import nn
from tqdm import tqdm

from config_train import data_info, train_opt
from data_read import datacreate
from Log import Logger
from lossfunction import CUPIDLoss
from node_creat import detemine_model, determine_data
from util import dict2str


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_value = train_opt.get('seed_value', 2024)
set_random_seed(seed_value)

ATIME = datetime.now().strftime('%Y%m%d_%H%M%S')
SDR = train_opt['savedir'] + 't' + ATIME + 'xx'
os.makedirs(SDR, exist_ok=True)
log = Logger(SDR + '/train.log', level='info')


class Node_model():
    def __init__(self, savedir):
        super(Node_model, self).__init__()
        self.mainnet = detemine_model()
        self.savedir = savedir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def e_save(self, sp):
        torch.save(self.mainnet.state_dict(), sp)

    def e_load(self, loaddir, flag='main'):
        log.logger.info('---------------------------------load pretrained model---------------------------')
        log.logger.info('load path:' + loaddir)

        if not loaddir:
            log.logger.info('empty load path, skip loading')
            return

        self.mainnet.to(self.device)
        pr_dict = torch.load(loaddir, map_location=self.device)
        if isinstance(pr_dict, dict) and 'state_dict' in pr_dict:
            pr_dict = pr_dict['state_dict']

        model_dict = self.mainnet.state_dict()
        pretrained_dict = {}
        for key, value in pr_dict.items():
            candidate_keys = [key]
            if not key.startswith('net.'):
                candidate_keys.append('net.' + key)

            for candidate_key in candidate_keys:
                if flag == 'in' and candidate_key.startswith('net.autoencoder.'):
                    continue
                if candidate_key in model_dict and model_dict[candidate_key].shape == value.shape:
                    pretrained_dict[candidate_key] = value
                    break

        log.logger.info('pretrained_dict len:{}------------------'.format(len(pretrained_dict)))
        model_dict.update(pretrained_dict)
        self.mainnet.load_state_dict(model_dict)

    def main_train(self, dataset, insertflag=False):
        self.mainnet.to(self.device)

        if train_opt['pretrainflag']:
            self.e_load(train_opt['pretrain_savedir'], flag='main')

        optimizer = optim.Adam(self.mainnet.parameters(), lr=train_opt['studyratio'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_opt['studyratio_ep'], gamma=train_opt['studyratio_de']
        )
        loss_function = nn.CrossEntropyLoss()
        train_loader = datacreate(dataset, flag='train', batch_size=train_opt['batchsize'])

        log.logger.info('-------------------------------------------------')
        log.logger.info('main train begin')

        for epoch in range(train_opt['epoch']):
            self.mainnet.train()
            train_bar = tqdm(train_loader)
            running_loss = 0.0
            running_acc = 0.0

            for step, data in enumerate(train_bar):
                feature, label = data
                feature = feature.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.mainnet(feature, insertflag)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, label)
                loss.backward()
                optimizer.step()

                batch_acc = torch.sum(preds == label.data).item() / label.size(0)
                running_loss = (running_loss * step + loss.item()) / (step + 1)
                running_acc = (running_acc * step + batch_acc) / (step + 1)
                train_bar.desc = 'train epoch[{}/{}] lr:{:.7f} loss:{:.3f} acc:{:.3f}'.format(
                    epoch + 1, train_opt['epoch'], optimizer.param_groups[0]['lr'], running_loss, running_acc
                )

            scheduler.step()
            log.logger.info(train_bar.desc)

        self.e_save(os.path.join(self.savedir, train_opt['savename'] + '_main'))
        self.main_val(dataset)

    def main_val(self, dataset, preflag=False):
        self.mainnet.to(self.device)
        if preflag:
            self.e_load(train_opt['pretrain_savedir'], flag='val')

        val_loader = datacreate(dataset, flag='validation')
        log.logger.info('-------------------------------------------------')
        log.logger.info('main validation begin')

        labellist, predlist = [], []
        self.mainnet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for feature, label in val_bar:
                feature = feature.to(self.device)
                outputs, _ = self.mainnet(feature)
                _, preds = torch.max(outputs, 1)
                labellist.extend(label.cpu().numpy().tolist())
                predlist.extend(preds.cpu().numpy().tolist())

        log.logger.info(classification_report(labellist, predlist, digits=4, zero_division=0))

    def in_train(self, dataset, insertflag=True):
        self.mainnet.to(self.device)

        if train_opt['i_pretrainflag']:
            self.e_load(train_opt['i_pretrain_savedir'], flag='in')
        elif train_opt['pretrainflag']:
            self.e_load(train_opt['pretrain_savedir'], flag='in')

        for module in self.mainnet.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

        for param in self.mainnet.net.autoencoder.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(self.mainnet.net.autoencoder.parameters(), lr=train_opt['i_studyratio'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_opt['i_studyratio_ep'], gamma=train_opt['i_studyratio_de']
        )
        loss_function = CUPIDLoss()
        train_loader = datacreate(dataset, flag='train', batch_size=train_opt['i_batchsize'])

        log.logger.info('-------------------------------------------------')
        log.logger.info('CUPID insert train begin')
        best_acc = -1.0

        for epoch in range(train_opt['i_epoch']):
            self.mainnet.eval()
            self.mainnet.net.autoencoder.train()
            train_bar = tqdm(train_loader)
            running_loss = 0.0
            running_acc = 0.0
            new_running_acc = 0.0
            detail_l = [0.0, 0.0]

            for step, data in enumerate(train_bar):
                feature, label = data
                feature = feature.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                outputs0, _ = self.mainnet(feature)
                newlabel = outputs0.detach()
                outputs, other_output = self.mainnet(feature, insertflag)
                _, preds = torch.max(outputs0, 1)
                _, newpreds = torch.max(outputs, 1)

                loss, detail_l = loss_function(
                    outputs,
                    other_output[0],
                    newlabel,
                    label,
                    other_output[1],
                    other_output[2],
                    train_opt['i_loss_para1'],
                    train_opt['i_loss_para2'],
                    train_opt['i_loss_para3'],
                )
                loss.backward()
                optimizer.step()

                batch_acc = torch.sum(preds == label.data).item() / label.size(0)
                new_batch_acc = torch.sum(newpreds == label.data).item() / label.size(0)
                running_loss = (running_loss * step + loss.item()) / (step + 1)
                running_acc = (running_acc * step + batch_acc) / (step + 1)
                new_running_acc = (new_running_acc * step + new_batch_acc) / (step + 1)

                train_bar.desc = (
                    'train epoch[{}/{}] lr:{:.7f} loss:{:.3f} loss_1:{:.3f} '
                    'loss_2:{:.3f} acc:{:.3f} newacc:{:.3f}'
                ).format(
                    epoch + 1,
                    train_opt['i_epoch'],
                    optimizer.param_groups[0]['lr'],
                    running_loss,
                    detail_l[0].detach().item(),
                    detail_l[1].detach().item(),
                    running_acc,
                    new_running_acc,
                )

            scheduler.step()
            log.logger.info(train_bar.desc)
            if epoch % train_opt['i_saveepoch'] == 0 and new_running_acc > best_acc:
                best_acc = new_running_acc
                self.e_save(os.path.join(self.savedir, train_opt['savename'] + '_insert'))
                log.logger.info('save model in {} epoch'.format(epoch))

        self.e_save(os.path.join(self.savedir, train_opt['savename'] + 'last_insert'))
        self.in_val(dataset)

    def in_val(self, dataset, preflag=False, insertflag=True, dataflag='validation'):
        self.mainnet.to(self.device)
        if preflag:
            self.e_load(train_opt['i_pretrain_savedir'], flag='val')

        val_loader = datacreate(dataset, flag=dataflag)
        log.logger.info('-------------------------------------------------')
        log.logger.info('CUPID insert validation begin')

        labellist, predlist, predlistinit = [], [], []
        softreslist, varusum = [], []

        self.mainnet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for feature, label in val_bar:
                feature = feature.to(self.device)
                outinit, _ = self.mainnet(feature)
                outlast, edresult = self.mainnet(feature, insertflag)
                _, preds = torch.max(outlast, 1)
                _, predsinit = torch.max(outinit, 1)

                softres = torch.abs(torch.softmax(outlast, dim=1) - torch.softmax(outinit, dim=1)).mean()
                sigma = edresult[0][0]

                labellist.extend(label.cpu().numpy().tolist())
                predlist.extend(preds.cpu().numpy().tolist())
                predlistinit.extend(predsinit.cpu().numpy().tolist())
                softreslist.append(float(softres.cpu()))
                varusum.append(float(torch.exp(sigma).mean().cpu()))

        log.logger.info('result initial ***************************')
        log.logger.info(classification_report(labellist, predlistinit, digits=4, zero_division=0))
        log.logger.info('result after insert ***************************')
        log.logger.info(classification_report(labellist, predlist, digits=4, zero_division=0))

        log.logger.info(
            'softres:{:.6f} var_usum:{:.6f}'.format(
                float(np.mean(softreslist)) if softreslist else 0.0,
                float(np.mean(varusum)) if varusum else 0.0,
            )
        )


def train_only_main(savedirroot):
    log.logger.info('++++++ train main ++++++')
    dataset = determine_data()
    savedir = savedirroot + '/main_model'
    os.makedirs(savedir, exist_ok=True)
    shutil.copy('config_train.py', savedir)
    net = Node_model(savedir)
    net.main_train(dataset)


def train_include_insert(savedirroot):
    log.logger.info('++++++ train CUPID insert ++++++')
    dataset = determine_data()
    savedir = savedirroot + '/in_model'
    os.makedirs(savedir, exist_ok=True)
    shutil.copy('config_train.py', savedir)
    if train_opt['i_pretrainflag'] and not train_opt['i_pretrain_savedir']:
        train_opt['i_pretrain_savedir'] = os.path.join(
            savedirroot, 'main_model', train_opt['savename'] + '_main'
        )
    net = Node_model(savedir)
    net.in_train(dataset)


if __name__ == '__main__':
    log.logger.info(dict2str(data_info))
    log.logger.info(dict2str(train_opt))

    train_only_main(SDR)
    train_include_insert(SDR)
