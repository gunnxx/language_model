import argparse
import logging
import os


parser = argparse.ArgumentParser()
#parser.add_argument('--')


def evaluate(model, optimizer, loss_fn, val_iterator, params):
    """Evaluate the model on evaluation data

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        val_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    total_loss = 0
    for batch in val_iterator:

        # compute model output and loss
        output = model(batch.input.to(params.device))
        loss = loss_fn(output, batch.target.to(params.device).long())
        total_loss = total_loss + loss

    logging.info("- Evaluation entropy loss: " + str(total_loss.item()))
    return total_loss