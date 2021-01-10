#import .config as config
from __future__ import absolute_import

import sys
import os

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.metric import NumpyMetric

try:
    from dotenv import find_dotenv, load_dotenv
    import wandb
    load_dotenv(find_dotenv())
    wandb.login(key=os.environ['WANDB_API_KEY'])
    from wandb.keras import WandbCallback
    _has_wandb = True
except:
    _has_wandb = False

import json

def BCEWithLogitsLoss(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

class PLF1Score(NumpyMetric):
    def __init__(self):
        super(PLF1Score, self).__init__(f1_score)
        self.scorer = f1_score

    def forward(self, x, y):
        x = np.round(np.array(x))
        y = np.round(np.array(y))

        return self.scorer(x,y)

class PLAccuracy(NumpyMetric):
    def __init__(self):
        super(PLAccuracy, self).__init__(accuracy_score)
        self.scorer = accuracy_score

    def forward(self, x, y):
        x = np.round(np.array(x))
        y = np.round(np.array(y))

        return self.scorer(x,y)

def test_pl_trainer(data_loader, model):
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = model(d['ids'], d['mask'], d['token_type_ids'])
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
    return fin_outputs

def test_torch(data_loader, model, device):
    fin_targets = []
    fin_outputs = []
    
    model.to(device)

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids'].to(device)
            mask = d['mask'].to(device)
            token_type_ids = d['token_type_ids'].to(device)

            outputs = model(ids, mask, token_type_ids)

            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
    return fin_outputs

class BasicTrainer:
    
    def __init__(self, model, train_data_loader, val_data_loader, device, test_data_loader=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.test_data_loader = test_data_loader
        self.model.to(self.device)

        self.print_stats()

    def print_stats(self):
        print ("[LOG] Total number of parameters to learn {}".format(sum(p.numel() for p in self.model.parameters() \
                                                                 if p.requires_grad)))

    def train_fn(self, data_loader):
        self.model.train()

        total_loss = 0
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for bi, d in pbar:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = self.loss_fn(outputs, targets)

            L2_reg = torch.tensor(0., requires_grad=True)
            for name, param in self.model.named_parameters():
                if 'weight' in name and 'attention' not in name:
                    L2_reg = L2_reg + torch.norm(param, 2)

            loss += self.l2 * L2_reg/len(data_loader)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss
            with np.printoptions(precision=3):
                v = round(loss.cpu().detach().numpy().item(), 3)
                pbar.set_description("Current training Loss {}".format(v))

        return total_loss

    def eval_fn(self, data_loader):
        self.model.eval()
        fin_targets = []
        fin_outputs = []

        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
            for bi, d in pbar:
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                targets = d["targets"]

                ids = ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.loss_fn(outputs, targets)

                L2_reg = torch.tensor(0., requires_grad=True)
                for name, param in self.model.named_parameters():
                    if 'weight' in name and 'attention' not in name:
                        L2_reg = L2_reg + torch.norm(param, 2)

                loss += self.l2 * L2_reg/len(data_loader)

                total_loss += loss

                with np.printoptions(precision=3):
                    v = round(loss.cpu().detach().numpy().item(), 3)
                    pbar.set_description("Current eval Loss {}".format(v))

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                
        return fin_outputs, total_loss, fin_targets

    def train(self, epochs, optimizer, scheduler, MODEL_PATH, config, l2=0, early_stopping_rounds=5, use_wandb=True, seed=42):

        self.scorer = f1_score #PLF1Score()
        self.loss_fn = BCEWithLogitsLoss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.l2 = l2
        self.args = config

        if use_wandb:
            wandb.init(project="wnut-task2-regularization",config=self.args)
            wandb.watch(self.model)
        else:
            with open(os.path.join(MODEL_PATH, 'parameters.json'),'w') as outfile:
                json.dump(self.args, outfile)

        if os.path.exists(os.path.join(MODEL_PATH,'model.bin')):
            try:
                try:
                    self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'model.bin')))
                except:
                    self.model = torch.load(os.path.join(MODEL_PATH,'model.bin'))

                print ("Loaded model from previous checkpoint")
            except:
                pass

        best_metric = 0
        bad_epochs = 0

        stats = {}
        for epoch in range(epochs):
            if bad_epochs < early_stopping_rounds:
                train_loss = self.train_fn(self.train_data_loader)

                print ("Running evaluation on whole training data")
                train_out, train_loss, train_targets = self.eval_fn(self.train_data_loader)
                print ("Running evaluation on validation data")
                val_out, val_loss, val_targets = self.eval_fn(self.val_data_loader)
                
                train_loss = train_loss/len(self.train_data_loader)
                val_loss = val_loss/len(self.val_data_loader)

                train_metric = self.scorer(np.round(np.array(train_targets)), np.round(np.array(train_out)))
                val_metric = self.scorer(np.round(np.array(val_targets)), np.round(np.array(val_out)))

                with np.printoptions(precision=3):
                    print("Train loss = {} Train metric = {} Val loss = {} Val metric = {}".format(round(train_loss.detach().cpu().numpy().item(), 3),round(train_metric,3),\
                        round(val_loss.detach().cpu().numpy().item(), 3),round(val_metric, 3)))
                
                if val_metric > best_metric:
                    torch.save(self.model.state_dict(), os.path.join(MODEL_PATH,'model.bin'))
                    print ("Saving best model in {}".format(MODEL_PATH))
                    #torch.save(self.model, os.path.join(MODEL_PATH,'model.bin'))
                    best_metric = val_metric
                    bad_epochs = 0
                else:
                    bad_epochs += 1

            else:
                print ("Early stopping")
                break

            stats.update({"epoch_{}".format(epoch): {"train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3),  "val_metric": round(val_metric,3)}})

            if use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric})

        if self.test_data_loader:
            self.test_output = test_fn(self.test_data_loader, self.model, self.device, self.final_activation)
        else:
            self.test_output = []

        #with np.printoptions(precision=3):
        #    print (stats)

        if use_wandb == False:
            with open(os.path.join(MODEL_PATH,'all_stats.json'), 'w') as outfile:
                json.dump(stats, outfile)

            with open(os.path.join(MODEL_PATH,'final_stats.json'), 'w') as outfile:
                d = {"epoch": epoch, "train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3),  "val_metric": round(val_metric,3)}
                json.dump(d, outfile)

class PLTrainer(pl.LightningModule):

    def __init__(self, num_train_steps, model, lr, l2=0, seed=42):
        super(PLTrainer, self).__init__()

        seed_everything(seed)

        self.model = model
        self.num_train_steps = num_train_steps
        self.lr = lr
        
        self.l2 = l2

        self.loss_fn = BCEWithLogitsLoss
        self.metric_name = 'f1'
        self.metric = PLAccuracy()

        self.save_hyperparameters()

        self.print_stats()
        
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        
    def print_stats(self):
        print ("[LOG] Total number of parameters to learn {}".format(sum(p.numel() for p in self.model.parameters() \
                                                                 if p.requires_grad)))

    def forward(self, x):

        return self.model(ids=x["ids"], mask=x["mask"], token_type_ids=x["token_type_ids"])
        #return self.model(ids=x["ids"], mask=x["mask"])

    def training_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        #outputs = self.model(ids=ids, mask=mask)

        loss = self.loss_fn(outputs, targets)

        L2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'attention' not in name:
                L2_reg = L2_reg + torch.norm(param, 2)

        loss = loss + self.l2 * L2_reg

        metric_value = self.metric(targets, torch.sigmoid(outputs))

        tensorboard_logs = {'train_loss': loss, "train {}".format(self.metric_name): metric_value}

        return {'loss': loss, 'train_metric': metric_value, \
                'log': tensorboard_logs, 'targets': targets, 'predictions': torch.sigmoid(outputs)}
    

    def validation_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        #outputs = self.model(ids=ids, mask=mask)

        loss = self.loss_fn(outputs, targets)

        L2_reg = torch.tensor(0., requires_grad=False)
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'attention' not in name:
                L2_reg = L2_reg + torch.norm(param, 2)

        loss = loss + self.l2 * L2_reg
        
        metric_value = self.metric(targets,torch.sigmoid(outputs))

        tensorboard_logs = {'val_loss': loss, "val {}".format(self.metric_name): metric_value}

        return {'val_loss': loss, 'val_metric': metric_value, \
                'log': tensorboard_logs, 'val_targets': targets, 'val_predictions': torch.sigmoid(outputs)}

    def configure_optimizers(self):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps)

        return [optimizer], [{'scheduler': scheduler}]

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        targets = []
        predictions = []
        for out in outputs:
            targets.extend(out['targets'])
            predictions.extend(out['predictions'])
        
        targets = torch.cat(targets, 0)
        predictions = torch.cat(predictions, 0)
        
        self.train_predictions.append(predictions)
        
        train_metric = torch.stack([x['train_metric'] for x in outputs]).mean()
        #train_metric = torch.tensor(np.array([self.metric(targets, predictions)]), dtype=torch.float)
        
        print ("Train loss = {} Train metric = {}".format(round(train_loss_mean.detach().cpu().numpy().item(), 3),round(train_metric.detach().cpu().numpy().item(), 3)))

        return {'train_loss': train_loss_mean, 'train_metric': train_metric}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        targets = []
        predictions = []
        for out in outputs:
            targets.extend(out['val_targets'])
            predictions.extend(out['val_predictions'])
        
        targets = torch.cat(targets, 0)
        predictions = torch.cat(predictions, 0)
        
        self.val_predictions.append(predictions)
        
        val_metric = torch.stack([x['val_metric'] for x in outputs]).mean()
        #val_metric = torch.tensor(np.array([self.metric(targets, predictions)]), dtype=torch.float)
        
        print ("val loss = {} val metric = {} ".format(round(val_loss_mean.detach().cpu().numpy().item(), 3),round(val_metric.detach().cpu().numpy().item(), 3)))

        return {'val_loss': val_loss_mean, 'val_metric': val_metric}

    