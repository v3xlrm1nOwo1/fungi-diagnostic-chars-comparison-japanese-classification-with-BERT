import torch
import config
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TextcDataset(Dataset):

  def __init__(self, data: pd.DataFrame, tokenizer: config.TOKENIZER, max_len: int = 512, include_row_text=False):
    self.data = data
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.include_row_text = include_row_text

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    data_row = self.data.iloc[item]

    sentence_text = str(data_row.sentence)
    labels = data_row['label_number']

    encoding = self.tokenizer.encode_plus(
        sentence_text,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True,
        return_attention_mask=True,
        max_length=self.max_len,
        padding='max_length',
        return_tensors='pt'
    )


    output =  {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

    if self.include_row_text:
        output['sentence_text'] = sentence_text

    return output



def create_data_loader(df, tokenizer, max_len=config.MAX_LEN, batch_size=config.TRAIN_BATCH_SIZE, include_row_text=False, shuffle=False):
  ds = TextcDataset(
    data=df,
    tokenizer=tokenizer,
    max_len=max_len,
    include_row_text=include_row_text
  )

  return DataLoader(
    ds,
    shuffle=shuffle,
    batch_size=batch_size,
    num_workers=4,
  )
  