import transformers
import config
import engine
from collections import defaultdict


def train(model, delta_model, train_data_loader, optimizer, val_data_loader, n_train_examples, n_val_examples):

    total_steps = len(train_data_loader) * config.NUM_EPOCHS

    scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(config.NUM_EPOCHS):

        print('=' * 50)
        print(f'Epoch {epoch + 1} / {config.NUM_EPOCHS}')

        train_acc, train_loss = engine.train_epoch(model=model, device=config.DEVICE, data_loader=train_data_loader, optimizer=optimizer, scheduler=scheduler, n_examples=n_train_examples)
        print(f'Train Loss: {train_loss}, Accuracy: {train_acc}')

        val_acc, val_loss = engine.eval_model(model=model, device=config.DEVICE, data_loader=val_data_loader, n_examples=n_val_examples)
        print(f'Val Loss: {val_loss}, Accuracy: {val_acc}')

        print(f'===> Epoch {epoch + 1} / {config.NUM_EPOCHS} || Train Loss: {train_loss}, Accuracy: {train_acc} || Val Loss: {val_loss}, Accuracy: {val_acc}')
        print('=' * 50)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            delta_model.save_finetuned('bert_model_state')
            best_accuracy = val_acc
            
    return best_accuracy, history
