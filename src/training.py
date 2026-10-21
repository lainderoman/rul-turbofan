"""Training loop with early stopping for the PyTorch models."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    """Fix all random seeds so that training gives the same result every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs=60,
    batch_size=256,
    learning_rate=1e-3,
    patience=10,
    seed=42,
    verbose=True,
):
    """Train a model with the Adam optimizer, MSE loss and early stopping.

    Early stopping: after every epoch we compute the loss on the validation
    set. If it has not improved for `patience` epochs in a row, we stop, and
    we put back the weights of the best epoch. This avoids overfitting and
    saves training time.

    Returns
    -------
    model : the model with the weights of the best validation epoch
    history : dict with the lists "train_loss" and "val_loss" (one value per epoch)

    Set verbose=False to hide the per-epoch messages.
    """
    set_seed(seed)

    # Wrap the numpy arrays in a DataLoader that gives shuffled mini-batches
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # ---- Training: one pass over all training windows ----
        model.train()  # enables dropout
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()                # reset gradients from the previous batch
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()                      # compute gradients
            optimizer.step()                     # update the weights
            total_loss += loss.item() * len(X_batch)
        train_loss = total_loss / len(train_dataset)

        # ---- Validation: no gradients, no dropout ----
        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(X_val_tensor), y_val_tensor).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping: no improvement for {patience} epochs.")
            break

    # Restore the weights of the best epoch
    model.load_state_dict(best_weights)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    if verbose:
        print(f"Best epoch: {best_epoch} (val loss {best_val_loss:.4f})")
    return model, history


def predict(model, X):
    """Predict with a trained model on a numpy array of windows."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()
