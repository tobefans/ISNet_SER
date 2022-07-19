# -*- coding: utf-8 -*-
"""
@author: fan weiquan
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import time
import random
import os
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.backends import cudnn
from dataset import Custom_dataset as Dataset
from network import Encoder, Translator, Classifier
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# system setting
plt.ion()
np.set_printoptions(suppress=True)

path_features = '/home/fwq/dataset/LSSED/spec_librosa/'
path_labels = '/home/fwq/dataset/LSSED/metadata_4class.csv'
path_model = 'model/'
path_result = 'result/'
path_pred = 'pred/'
path_pic = 'picture/'

for dirs in [path_model, path_result, path_pred, path_pic]:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 32
params_E = {'lr': 1e-2, 'step_size': 20, 'gamma': 1e-1}
params_C = {'lr': 1e-2, 'step_size': 20, 'gamma': 1e-1}
params_T = {'lr': 1e-2, 'step_size': 20, 'gamma': 1e-1}
params_E2 = {'lr': 1e-2, 'step_size': 20, 'gamma': 1e-1}
params_C2 = {'lr': 1e-2, 'step_size': 20, 'gamma': 1e-1}
use_GPU = True
WD = 1e-5
num_classes = 4
num_pairs = 3
mode = 'base'
mode = 'isnet'
domain = 'lssed_isnet'
print(domain)

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

## import data
path_metadata = 'metadata_pairs.csv'
if os.path.exists(path_metadata):
    df = pd.read_csv(path_metadata)
else:
    df = pd.read_csv(path_labels)
    df = df[df.is_Chinese == False].reset_index(drop=True)
    df['speakers'] = [int(s[1:5]) for s in df.name]
    print('Length: ', len(df))
    ## search pairs
    name_pair = []
    spk2pairs = dict()
    for i in range(len(df)):
        if df.label[i] == 1:
            name_pair.append(df.name[i])
        else:
            if df.speakers[i] not in spk2pairs:
                df_tmp = df[df.speakers[i]==df.speakers]
                try:
                    pairs = df_tmp[df_tmp.label==1].sample(num_pairs).name.values
                    name_pair.append('-'.join(pairs))
                    spk2pairs[df.speakers[i]] = '-'.join(pairs)
                except:
                    print('No Found: ', df.name[i])
                    df = df.drop(i)
            else:
                name_pair.append(spk2pairs[df.speakers[i]])

    df['name_pair'] = name_pair
    df = df.reset_index(drop=True)
    print('Length: ', len(df))
    df.to_csv(path_metadata, index=None)
train_names_labels_genders = df[df.speakers < 883]
test_names_labels_genders = df[df.speakers >=883]

num = np.array([sum(df['label']==i) for i in range(num_classes)])
weights = np.around(min(num) / num, 3)

print('Num: Train: ', np.array([sum(train_names_labels_genders['label']==i) for i in range(num_classes)]))
print('Num: Test:  ', np.array([sum(test_names_labels_genders['label']==i) for i in range(num_classes)]))

## balance
num = np.array([sum(train_names_labels_genders['label']==i) for i in range(num_classes)])
# ind_bal = [np.random.randint(0, num[i], max(num)-num[i]) for i in range(num_classes)]
ind_bal = [np.random.choice(num[i], num[i]-sorted(num)[-2], replace=False) if sorted(num)[-2]-num[i]<0 else np.random.randint(0, num[i], sorted(num)[-2]-num[i]) for i in range(num_classes)]
flag_bal = [sorted(num)[-2]-num[i]<0 for i in range(num_classes)]
for i in range(num_classes):
    if not flag_bal[i]:
        train_names_labels_genders = train_names_labels_genders.append(train_names_labels_genders[train_names_labels_genders['label']==i].iloc[ind_bal[i], :])
print('Bal Num: Train: ', np.array([sum(train_names_labels_genders['label']==i) for i in range(num_classes)]))


datasets = {'train': Dataset(path_features, train_names_labels_genders, num_pairs),
            'test' : Dataset(path_features, test_names_labels_genders,  num_pairs)}
loaders = {'train': Data.DataLoader(dataset=datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
            'test': Data.DataLoader(dataset=datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)}

E = nn.DataParallel(Encoder()).to(device)
C = nn.DataParallel(Classifier()).to(device)
if mode == 'isnet':
    E2 = nn.DataParallel(Encoder()).to(device)
    C2 = nn.DataParallel(Classifier()).to(device)
    T = nn.DataParallel(Translator()).to(device)

opt_E = torch.optim.SGD(E.parameters(), lr=params_E['lr'], weight_decay=WD, momentum=0.9)
opt_C = torch.optim.SGD(C.parameters(), lr=params_C['lr'], weight_decay=WD, momentum=0.9)
scheduler_E = lr_scheduler.StepLR(opt_E, step_size=params_E['step_size'], gamma=params_E['gamma'])
scheduler_C = lr_scheduler.StepLR(opt_C, step_size=params_C['step_size'], gamma=params_C['gamma'])
if mode == 'isnet':
    opt_E2 = torch.optim.SGD(E2.parameters(), lr=params_E2['lr'], weight_decay=WD, momentum=0.9)
    scheduler_E2 = lr_scheduler.StepLR(opt_E2, step_size=params_E2['step_size'], gamma=params_E2['gamma'])
    opt_C2 = torch.optim.SGD(C2.parameters(), lr=params_C2['lr'], weight_decay=WD, momentum=0.9)
    opt_T = torch.optim.SGD(T.parameters(), lr=params_T['lr'], weight_decay=WD, momentum=0.9)
    scheduler_C2 = lr_scheduler.StepLR(opt_C2, step_size=params_C2['step_size'], gamma=params_C2['gamma'])
    scheduler_T = lr_scheduler.StepLR(opt_T, step_size=params_T['step_size'], gamma=params_T['gamma'])

loss_class = nn.CrossEntropyLoss().to(device)
loss_dist = nn.L1Loss().to(device)


loss_train, loss_test = [], []
loss2_train, loss2_test = [], []
metric_train, metric_test = [], []
accuracy_train, accuracy_test = [], []
m_was, m_uas, s_was, s_uas = [], [], [], []

best_acc = -1000.0
index_epoch = 0
for epoch in range(EPOCH):
# -----------------------train------------------------------------------------
    E.train()
    C.train()
    if mode == 'isnet':
        E2.train()
        C2.train()
        T.train()
    loss_tr, loss2_tr = 0.0, 0.0
    pred_all, actu_all = [], []
    start_time = time.time()
    for step, (xs, xs_pairs, ys) in enumerate(loaders['train'], 0):
        xs, ys = xs.to(device), ys.to(device)

        ## Train E, C
        out, features = C(E(xs))
        loss = loss_class(out, ys)
        E.zero_grad()
        C.zero_grad()
        loss.backward()
        opt_E.step()
        opt_C.step()

        if mode == 'isnet':
            ## Train E
            out_E_pairs = []
            for xs_pair in xs_pairs:
                out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            loss3 = 0
            for i in range(num_pairs-1):
                for j in range(i+1, num_pairs):
                    loss3 += loss_dist(out_E_pairs[:,i], out_E_pairs[:,j])
            E2.zero_grad()
            loss3.backward()
            opt_E2.step()

            ## Train T
            out_E_pairs = []
            for xs_pair in xs_pairs:
                out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
            out_T = T(E(xs))
            loss2 = loss_dist(out_T, out_E_pair)
            T.zero_grad()
            loss2.backward()
            opt_T.step()

            ## Train C2
            out_E = E(xs)
            out_E = out_E - T(out_E)
            out, features = C2(out_E)
            loss = loss_class(out, ys)
            C2.zero_grad()
            loss.backward()
            opt_C2.step()

        else:
            loss2 = loss

        pred = torch.max(out.cpu().data,1)[1].numpy()
        actu = ys.cpu().data.numpy()
        pred_all = pred_all + list(pred)
        actu_all = actu_all + list(actu)

        loss_tr += loss.cpu().item()
        loss2_tr += loss2.cpu().item()
    loss_tr = loss_tr / len(loaders['train'].dataset)
    loss2_tr = loss2_tr / len(loaders['train'].dataset)


    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    metric_tr = metrics.recall_score(actu_all,pred_all,average='macro')
    accuracy_tr = metrics.accuracy_score(actu_all,pred_all)

    loss_train.append(loss_tr)
    loss2_train.append(loss2_tr)
    metric_train.append(metric_tr)
    accuracy_train.append(accuracy_tr)
    print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| metric: %.3f' % metric_tr, '| acc: %.3f' % accuracy_tr, '| lr: %.5f' % opt_E.param_groups[0]['lr'])  
    
    time_train = time.time() - start_time
    start_time = time.time()

# ----------------------test--------------------------------------------------
    with torch.no_grad():
        E.eval()
        C.eval()
        if mode == 'isnet':
            E2.eval()
            C2.eval()
            T.eval()

        loss_te, loss2_te = 0.0, 0.0
        pred_all, actu_all = [], []
        for step, (xs, xs_pairs, ys) in enumerate(loaders['test'], 0):
            xs, ys = xs.to(device), ys.to(device)

            out, _ = C(E(xs))
            loss = loss_class(out, ys)

            if mode == 'isnet':
                out_E_pairs = []
                for xs_pair in xs_pairs:
                    out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
                out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
                out_T = T(E(xs))
                loss2 = loss_dist(out_T, out_E_pair)

                out_E = E(xs)
                out_E = out_E - T(out_E)
                out, features = C2(out_E)
                loss = loss_class(out, ys)
            else:
                loss2 = loss

            pred = torch.max(out.cpu().data,1)[1].numpy()
            actu = ys.cpu().data.numpy()
            pred_all = pred_all + list(pred)
            actu_all = actu_all + list(actu)

            loss_te += loss.cpu().item()
            loss2_te += loss2.cpu().item()

        loss_te = loss_te / len(loaders['test'].dataset)
        loss2_te = loss2_te / len(loaders['test'].dataset)

        pred_all, actu_all = np.array(pred_all), np.array(actu_all)
        metric_te = metrics.recall_score(actu_all,pred_all,average='macro')
        accuracy_te = metrics.accuracy_score(actu_all,pred_all)
        
        loss_test.append(loss_te)
        loss2_test.append(loss2_te)
        metric_test.append(metric_te)
        accuracy_test.append(accuracy_te)
        print('TEST :: Epoch: ', epoch, '| Loss: %.3f' % loss_te, '| metric: %.3f' % metric_te, '| acc: %.3f' % accuracy_te, '| lr: %.5f' % opt_C.param_groups[0]['lr'])

        cm = metrics.confusion_matrix(actu_all, pred_all)
        cm = cm / np.sum(cm, axis=1).reshape((-1,1))
        print(np.around(cm, 4)*100)

        scheduler_E.step()
        scheduler_C.step()
        if mode == 'isnet':
            scheduler_E2.step()
            scheduler_C2.step()
            scheduler_T.step()

    time_epoch = time.time() - start_time
    print('Train: Epoch {:.0f} completed with {:.0f}m {:.0f}s'.format(epoch, time_train // 60, time_train % 60))
    print('Test:  Epoch {:.0f} completed with {:.0f}m {:.0f}s'.format(epoch, time_epoch // 60, time_epoch % 60))

    if accuracy_te + metric_te > best_acc:
        index_epoch = epoch
        best_acc = accuracy_te + metric_te
        pred_all_early = pred_all
        pd.DataFrame({'prediction':pred_all, 'label':actu_all}).to_csv(path_pred+domain+'_pred_test.csv', index=False, sep=',')
        if mode == 'isnet':
            torch.save({'E':E.state_dict(),'C':C.state_dict(),'E2':E2.state_dict(),'C2':C2.state_dict(),'T':T.state_dict()}, path_model+domain+'.pkl')
        else:
            torch.save({'E':E.state_dict(),'C':C.state_dict()}, path_model+domain+'.pkl')

dataframe = pd.DataFrame({'loss_train':loss_train, 'unaccuracy_train':metric_train, 'loss_test':loss_test,'unaccuracy_test':metric_test, 'accuracy_train':accuracy_train,'accuracy_test':accuracy_test})  
dataframe.to_csv(path_result+domain+'_train_test.csv', index=False, sep=',')

## plot
fig,axes=plt.subplots(2,4)
axes[0,0].plot(loss_train, color="blue", lw = 2.5, linestyle="-")
axes[0,1].plot(loss2_train, color="blue", lw = 2.5, linestyle="-")
axes[0,2].plot(accuracy_train, color="black", lw = 2.5, linestyle="-")
axes[0,3].plot(metric_train, color="black", lw = 2.5, linestyle="-")
axes[1,0].plot(loss_test, color="blue", lw = 2.5, linestyle="-")
axes[1,1].plot(loss2_test, color="blue", lw = 2.5, linestyle="-")
axes[1,2].plot(accuracy_test, color="black", lw = 2.5, linestyle="-")
axes[1,3].plot(metric_test, color="black", lw = 2.5, linestyle="-")
fig.savefig(path_pic + domain + '.png')


print('TEST:: metric: %.3f' % metric_test[index_epoch], '| acc: %.3f' % accuracy_test[index_epoch])
print('TEST:: m_wa: %.3f' % m_was[index_epoch], '| m_ua: %.3f' % m_uas[index_epoch], '| s_wa: %.3f' % s_was[index_epoch], '| s_ua: %.3f' % s_uas[index_epoch])
print(metrics.classification_report(actu_all, pred_all_early))
cm = metrics.confusion_matrix(actu_all, pred_all_early)
cm = cm / np.sum(cm, axis=1).reshape((-1,1))
print(np.around(cm, 4)*100)
print(index_epoch)
print(domain)
