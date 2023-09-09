import transformers, torch

MAX_LEN = 150
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
NUMBER_OF_CLASSES = 25
LEARN_RATE = 3e-6
RANDOM_SEED = 666
CHECKPOINT = 'cl-tohoku/bert-base-japanese-v3'
TOKENIZER = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
