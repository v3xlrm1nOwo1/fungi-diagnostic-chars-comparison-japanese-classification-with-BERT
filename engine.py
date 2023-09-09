import torch
import numpy as np
import torch.nn as nn


def train_epoch(model, device, data_loader, optimizer, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for index, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs, loss, accuracy = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
        )

        correct_predictions += accuracy
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

    return  float(correct_predictions) / n_examples, np.mean(losses)


def eval_model(model, device, data_loader, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for index, data in enumerate(data_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs, loss, accuracy = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
            )
            
            correct_predictions += accuracy
            losses.append(loss.item())

    return float(correct_predictions) / n_examples, np.mean(losses)
