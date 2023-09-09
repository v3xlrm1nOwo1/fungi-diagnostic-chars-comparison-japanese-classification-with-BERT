import torch
import config
import numpy as np
import opendelta
import torch.nn.functional as F


def prediction(test_data_loader, model, n_examples, device=config.DEVICE):
    losses = []
    correct_predictions = 0
    
    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    delta_model = opendelta.AutoDeltaModel.from_finetuned('/content/bert_model_state', backbone_model=model)

    test_data_loader_n = len(test_data_loader)

    model.eval()

    for index, batch in enumerate(test_data_loader):
        texts = batch["sentence_text"]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs, loss, accuracy = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
            )
            
            correct_predictions += accuracy
            losses.append(loss.item())

        _, preds = torch.max(outputs, dim=1)

        probs = F.softmax(outputs, dim=1)

        sentences.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(probs)
        real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    print("Prediction Done.....!!!")
    return float(correct_predictions) / n_examples, np.mean(losses), sentences, predictions, prediction_probs, real_values
