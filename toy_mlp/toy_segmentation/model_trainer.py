from typing import Tuple
from tqdm import tqdm
import numpy as np

from nn_lib import Tensor, SetGrad
from nn_lib.mdl import Module
from nn_lib.mdl.loss_functions import Loss
from nn_lib.optim import Optimizer
from nn_lib.data import Dataloader

from nn_lib.scheduler.scheduler import Scheduler
from nn_lib.scheduler.multi_step_lr import MultiStepLR

from nn_lib.support_func import iou_score
from matplotlib import pyplot as plt

class UNetTrainer(Module):
    """
    A helper class for manipulating a neural network training and validation
    """
    def __init__(self, model: Module, loss_function: Loss, optimizer: Optimizer, scheduler:Scheduler=None, score_function = None):
        """
        Create a neural network
        :param model: a model to train
        :param loss_function: loss function module to use for training
        :param optimizer: optimizer to use for training
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.score_function = iou_score
        if score_function:
            self.score_function = score_function

        self.train_loss = []
        self.val_loss = []
        self.train_score = []
        self.val_score = []
        self.train_dataloader: Dataloader
        self.val_dataloader: Dataloader

        if scheduler:
            self.scheduler = scheduler

    def set_datasets(self, train_dataloader: Dataloader,val_dataloader: Dataloader):
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply model.forward(x)
        :param x: input data batch of the shape (B, in_features)
        :return: prediction logits of the shape (B,)
        """
        predictions = self.model.forward(x)
        return predictions

    def train(self,n_epochs: int):
        progress_bar = tqdm(n_epochs)
        for i_epoch in range(n_epochs):
            self.train_loop()
            if len(self.train_loss) > 0 and len(self.train_score) > 0:
                progress_bar.desc = f'Training. Epoch: {i_epoch + 1}. Loss: {self.train_loss[-1]:.4f}. Score: {self.train_score[-1]:.4f}'
            data_batch, label_batch, prediction_logit_batch = self.val_loop()
            if len(self.val_loss) > 0 and len(self.val_score) > 0:
                progress_bar.desc = f'Validation. Epoch: {i_epoch + 1}. Loss: {self.val_loss[-1]:.4f}. Score: {self.val_score[-1]:.4f}'
            self.show_res(data_batch, label_batch, prediction_logit_batch,i_epoch)
            progress_bar.update(1)


    def train_loop(self) -> None:
        """
        Train a model on the given data for the given number of epochs
        :param train_dataloader: dataloader of training data
        :param n_epochs: number of epochs to train for
        :return: None
        """
        loss = 0
        score = 0
        n = 0
        for data_batch, label_batch in self.train_dataloader:
            _, loss_value = self._train_step(data_batch, label_batch)
            loss += loss_value.data.item()
            n += 1

        if self.score_function:
            pass  # some scoring stuff
        self.train_loss.append(loss / n)

        if self.scheduler:
            self.scheduler.step()


    def _train_step(self, data_batch: Tensor, label_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        A single training step that
            (1) performs model forward on the data batch
            (2) compares predictions to the batch of labels by computing loss
            (3) performs backward pass computing gradients of loss over model parameters and
            (4) updates model parameters based on the computed gradients
        :param data_batch: data batch to train on of the shape (B, in_features)
        :param label_batch: label batch corresponding to the data batch of the shape (B,)
        :return: tuple of two tensors for prediction logits and loss value
        """
        # Zero you grad for every batch
        self.optimizer.zero_grad()
        output = self.model.forward(data_batch)

        loss = self.loss_function.forward(output, label_batch)

        loss.backward()

        self.optimizer.step()

        return tuple([ output , loss ])

    def val_loop(self):
        """
        Validate the model on the test data
        :param test_dataloader: data to validate on
        :return: a tuple for binary predictions, accuracy of the model and mean loss of the model
        """
        SetGrad.disable_grad()

        loss_values_sum = 0.0
        n_predictions = 0
        score_value_sum = 0.0

        #for data_batch, label_batch in tqdm(self.val_dataloader, desc='Validating'):
        for data_batch, label_batch in self.val_dataloader:
            prediction_logit_batch = self.model.forward(data_batch)
            n_predictions += len(data_batch.data)
            loss_value = self.loss_function(prediction_logit_batch, label_batch)
            loss_values_sum += loss_value.data.item()

            if self.score_function:
                score_value_sum += self.score_function(prediction_logit_batch,label_batch).mean().item()


        self.val_loss.append(loss_values_sum / n_predictions)
        self.val_score.append(score_value_sum / n_predictions)

        SetGrad.enable_grad()

        return data_batch, label_batch, prediction_logit_batch

    def show_res(self, data_batch:Tensor, label_batch:Tensor, prediction_logit_batch:Tensor,epoch:int):
        # clear_output(wait=True)

        for k in range(6):
            plt.subplot(3, 6, k + 1)
            plt.imshow(data_batch.data[k])
            plt.title('Real'+str(epoch))
            plt.axis('off')

            plt.subplot(3, 6, k + 7)
            plt.imshow(prediction_logit_batch.data[k], cmap='gray')

            plt.title('Output')
            plt.axis('off')

            plt.subplot(3, 6, k + 13)
            plt.imshow(label_batch.data[k], cmap='gray')
            plt.title('Mask')
            plt.axis('off')

        # or plt.suptitle('Main title')

        plt.legend()
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.show()
        plt.cla()



    @property
    def history(self) -> Tuple:
        return self.train_loss, self.train_score, self.val_loss, self.val_score

    def predict(self, data):
        with SetGrad.disable_grad():
            Y_pred = [self.model.forward(X_batch) for X_batch, _ in data]
        return Y_pred

    def score_model(self, metric, data):
        scores = 0
        for X_batch, Y_label in data:
            with SetGrad.disable_grad():
                Y_pred = self.model.forward(X_batch)
                Y_pred = np.ones_like(Y_pred) * (Y_pred > Tensor(0.5,requires_grad=False))
                scores += self.loss_function(Y_pred, Y_label).mean().item()
        return scores / len(data)