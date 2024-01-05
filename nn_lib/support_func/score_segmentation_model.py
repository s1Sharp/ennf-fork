from nn_lib.grad_modifier import SetGrad

import numpy as np

def predict(model, data):
    with SetGrad.disable_grad():
      Y_pred = [model(X_batch) for X_batch, _ in data]
    return Y_pred

def score_model(model, metric, data):
    scores = 0
    for X_batch, Y_label in data:
      with SetGrad.disable_grad():
        Y_pred = model(X_batch)
        Y_pred = np.ones_like(Y_pred) * (Y_pred > 0.5)
        scores += metric(Y_pred, Y_label).mean().item()
    return scores/len(data)