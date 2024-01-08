from nn_lib.mdl.loss_functions import CELoss, BCELoss_logits
from nn_lib.optim import Adam, Optimizer
from nn_lib.data import Dataloader

from toy_mlp.toy_segmentation.model_trainer import UNetTrainer
from toy_mlp.mnist_classifier.mnist_mlp_classifier import MnistMLPClassifier
from toy_mlp.toy_segmentation.unet import UNet
from toy_mlp.toy_segmentation.small_unet import SmallUNet
from toy_mlp.toy_segmentation.small_unet_1 import SmallUNet1
from toy_mlp.history_plotter import plot_loss
from toy_mlp.toy_segmentation.PH2_dataset import PH2


from nn_lib.scheduler.multi_step_lr import MultiStepLR


def main(n_epochs, optim: Optimizer = Adam,batch_size=25, milestones=[]):
    # create binary MLP classification model
    mlp_model = SmallUNet1()
    print(f'Created the following binary MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = BCELoss_logits() #CELoss()
    # create optimizer for model parameters
    optimizer = optim(mlp_model.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    # create a model trainer
    model_trainer = UNetTrainer(mlp_model, loss_fn, optimizer, scheduler=scheduler)

    # generate a training dataset
    size = (16,16)
    #size = (32,32)
    train_dataset = PH2(ds_type='train',size=size)
    # generate a validation dataset different from the training dataset
    val_dataset = PH2(ds_type='val',size=size)
    # create a dataloader for training data with shuffling and dropping last batch
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # create a dataloader for validation dataset without shuffling or last batch dropping
    val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # train the model for a given number of epochs
    model_trainer.set_datasets(train_dataloader, val_dataloader)
    model_trainer.train(n_epochs)
    # validate model on the train data
    # Note: we create a new dataloader without shuffling or last batch dropping

    # plot_loss(model_trainer.history_loss)
    return model_trainer.history


if __name__ == '__main__':
    main(10,Adam,25,[0])