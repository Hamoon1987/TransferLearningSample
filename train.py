import os
from model import fetch_model
from dataloader import fetch_dataloader
from config import args
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Training:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter('runs')
        self.epochs = args.epochs
        self.loss = nn.CrossEntropyLoss()
        self.model = fetch_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        self.training_loader, self.validation_loader = fetch_dataloader()
        self.running_loss_history = []
        self.running_corrects_history = []
        self.val_running_loss_history = []
        self.val_running_corrects_history = []

    def save_model(self, e):
        torch.save({
                    'epoch': e,
                    'model_state_dict': self.model.cpu().state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join('save_model', 'epoch-{}.pth'.format(e)))

    def train(self):
        for e in range(self.epochs):

            running_loss = 0.0
            running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0
            
            for inputs, labels in self.training_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            else:
                with torch.no_grad():
                    for val_inputs, val_labels in self.validation_loader:
                        val_inputs = val_inputs.to(self.device)
                        val_labels = val_labels.to(self.device)
                        val_outputs = self.model(val_inputs)
                        val_loss = self.loss(val_outputs, val_labels)
                        
                        _, val_preds = torch.max(val_outputs, 1)
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(val_preds == val_labels.data)
                
                epoch_loss = running_loss/len(self.training_loader.dataset)
                epoch_acc = running_corrects.float()/ len(self.training_loader.dataset)
                self.running_loss_history.append(epoch_loss)
                self.running_corrects_history.append(epoch_acc)
                
                val_epoch_loss = val_running_loss/len(self.validation_loader.dataset)
                val_epoch_acc = val_running_corrects.float()/ len(self.validation_loader.dataset)
                self.val_running_loss_history.append(val_epoch_loss)
                self.val_running_corrects_history.append(val_epoch_acc)
                print('epoch :', (e+1))
                print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
                print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

                self.writer.add_scalar('training_loss', epoch_loss, e+1)
                self.writer.add_scalar('training_accuracy', epoch_acc, e+1)
                self.writer.add_scalar('validation_loss', val_epoch_loss, e+1)
                self.writer.add_scalar('validation_accuracy', val_epoch_acc, e+1)

                if e % 2 == 0:
                    self.save_model(e)
                    self.model.to(self.device)
                    last_saved_epoch = e
        if last_saved_epoch != e:
            self.save_model(e)
                
            
if __name__ == '__main__':
    train = Training()
    train.train()