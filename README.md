


### Fungi Diagnostic Chars Comparison Japanese With <a href='https://huggingface.co/cl-tohoku/bert-base-japanese-v3'>BERT Base Japanese</a>


### Model 
<a href='https://huggingface.co/cl-tohoku/bert-base-japanese-v3'>BERT Base Japanese v3</a> From <a href='https://huggingface.co/Atsushi'>Atsushi</a>

This version of the model processes input texts with word-level tokenization based on the Unidic 2.1.2 dictionary (available in unidic-lite package), followed by the WordPiece subword tokenization. Additionally, the model is trained with the whole word masking enabled for the masked language modeling (MLM) objective.

### Parameter Efficient Finetuning 
I used <a href='https://opendelta.readthedocs.io/en/latest/index.html#'>opendelta library</a> for parameter efficient finetuning

```zsh
> Adapter 
trainable params: 903744 - all params: 109216323 - trainable: 0.827481% 
```

### What is a Adapter

Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter's effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.
<a href='https://arxiv.org/abs/1902.00751'>Adapter Paper</a>

### Dataset

The Dataset is Fungi Diagnostic Chars Comparison Japanese.
<a href='https://huggingface.co/datasets/Atsushi/fungi_diagnostic_chars_comparison_japanese'>Dataset</a> From  <a href='https://huggingface.co/Atsushi'>Atsushi</a>

### Note

I did not have the resources, such as the Internet, electricity, device, etc., to train the model well and choose the appropriate learning rate, so there were no results.


> To contribute to the project, please contribute directly. I am happy to do so, and if you have any comments, advice, job opportunities, or want me to contribute to a project, please contact me <a href='mailto:V3xlrm1nOwo1@gmail.com' target='blank'>V3xlrm1nOwo1@gmail.com</a>
