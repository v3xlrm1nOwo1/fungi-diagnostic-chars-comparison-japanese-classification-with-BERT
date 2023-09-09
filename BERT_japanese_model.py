import config
import torch
import torch.nn as nn
import transformers


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels).to(config.DEVICE)

def accuracy_fn(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=config.NUMBER_OF_CLASSES):
        super(SentimentClassifier, self).__init__()
        self.n_classes = n_classes
        self.model = transformers.AutoModel.from_pretrained(config.CHECKPOINT)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, labels):
        pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.drop(pooled_output[1])
        output = self.out(output)
        
        loss = loss_fn(outputs=output, labels=labels)
        accuracy = accuracy_fn(outputs=output, labels=labels)

        return output, loss, accuracy
    
    