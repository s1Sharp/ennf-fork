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

class ModelTrainer(Module):
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
        self.score_function = None
        if score_function:
            self.score_function = score_function

        self.history_loss = []
        self.history_score = []

        if scheduler:
            self.scheduler = scheduler

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply model.forward(x)
        :param x: input data batch of the shape (B, in_features)
        :return: prediction logits of the shape (B,)
        """
        predictions = self.model.forward(x)
        return predictions

    def train(self, train_dataloader: Dataloader, n_epochs: int) -> None:
        """
        Train a model on the given data for the given number of epochs
        :param train_dataloader: dataloader of training data
        :param n_epochs: number of epochs to train for
        :return: None
        """
        progress_bar = tqdm(range(n_epochs * len(train_dataloader)))
        for i_epoch in range(n_epochs):
            loss = 0
            score = 0
            n = 0
            for data_batch, label_batch in train_dataloader:
                _, loss_value = self._train_step(data_batch, label_batch)
                progress_bar.update(1)
                progress_bar.desc = f'Training. Epoch: {i_epoch + 1}. Loss: {loss_value.data.item():.4f}'

                if self.score_function:
                    pass  # some scoring stuff
                loss += loss_value.data.item()
                n += 1

            self.history_loss.append(loss / n)



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

    def validate(self, test_dataloader: Dataloader,multiclass=False) -> Tuple[np.ndarray, float, float]:
        """
        Validate the model on the test data
        :param test_dataloader: data to validate on
        :return: a tuple for binary predictions, accuracy of the model and mean loss of the model
        """
        SetGrad.disable_grad()

        n_correct_predictions = 0
        n_predictions = 0
        loss_values_sum = 0
        predictions = []
        for data_batch, label_batch in tqdm(test_dataloader, desc='Validating'):
            prediction_logit_batch = self.model(data_batch)
            if multiclass:
                positive_predictions = np.argmax(prediction_logit_batch.data,axis=1)
                correct_predictions = positive_predictions == np.argmax(label_batch.data, axis=1)
                n_correct_predictions += correct_predictions.sum()
                n_predictions += len(data_batch.data)
            else:
                positive_predictions = prediction_logit_batch.data > 0
                correct_predictions = positive_predictions == label_batch.data
                n_correct_predictions += correct_predictions.sum()
                n_predictions += len(data_batch.data)

            loss_value = self.loss_function(prediction_logit_batch, label_batch)
            loss_values_sum += loss_value.data

            predictions.extend(positive_predictions.tolist())
        if not multiclass:
            predictions = np.array(predictions, bool)

        accuracy = n_correct_predictions / n_predictions
        mean_loss = loss_values_sum / n_predictions

        SetGrad.enable_grad()

        return predictions, accuracy, mean_loss

    @property
    def history(self) -> list[int]:
        return self.history_loss