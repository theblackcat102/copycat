# Reward Model pretrained on openai/webgpt_comparison

Reward model finetuned from existing pretrain model.

Things that aligned with the orignal papers

* Overfits easily using rank loss

* Small learning rate

Different from the papers


* Small model performs bad due to lack of world knowledge, since the validation accuracy doesn't even reach 60%. OpenAI RM had 6B parameters.

* Train using a 80-20 train-validation split on torch AMP settings

* Added negative samples from flan-t5-base and flan-t5-large. This ensures the model will rank smaller models with lower score. Otherwise I find "small" model such as electra-large will rank flan-t5-base higher than the positive answer

Other models I had tried

* bloomz-560m : embedding size doesn't worth the training, since this dataset only contain english prompt

* gpt2-large : not stable 

* gpt2-base : not stable


# Performance on validation split

| model  | val acc  | val loss (rank loss)  |
|---|---|---|
| [roberta-base](https://huggingface.co/theblackcat102/roberta-base-webgpt-rm)  | 56.21  |  0.71 |
| [roberta-large](https://huggingface.co/theblackcat102/roberta-large-webgpt-rm)  | 57.89  |  0.67 |
| [electra-base](https://huggingface.co/theblackcat102/electra-base-webgpt-rm)  | 57.02  | 0.70  |
| [electra-large](https://huggingface.co/theblackcat102/electra-large-webgpt-rm)  | 58.75  | 0.69  |

Tensorboard logs are located under runs/


# Training

```
python train_critic.py
```

# Note:

* You will have to reweight this model output such that the mean rewards equals to 0
