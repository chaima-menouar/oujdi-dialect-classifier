base_model: distilbert-base-multilingual-cased
library_name: peft
pipeline_tag: text-classification
language:

* ar
  tags:
* darija
* moroccan-arabic
* oujdi
* text-classification
* transformers
* lora
* peft



# Oujdi Dialect Classification Model

This model classifies short text as either Oujdi, the regional dialect spoken in Eastern Morocco, or general Moroccan Darija.

It is a LoRA adapter fine-tuned on top of `distilbert-base-multilingual-cased` for binary sequence classification. The model supports Arabic, Latin, and mixed-script text.

## Labels

| Label ID | Class           |
| -------: | --------------- |
|        0 | Moroccan Darija |
|        1 | Oujdi           |

## Dataset

The model was trained on a balanced dataset containing 100,000 text samples:

| Class           | Samples |
| --------------- | ------: |
| Oujdi           |  50,000 |
| Moroccan Darija |  50,000 |

The Moroccan Darija samples were prepared from the Darija Open Dataset. The Oujdi samples were prepared from a 220,000-line corpus created by the project team.

The final dataset was divided into:

* 80% training data
* 10% validation data
* 10% test data

## Training Configuration

| Parameter               | Value                                |
| ----------------------- | ------------------------------------ |
| Base model              | `distilbert-base-multilingual-cased` |
| LoRA rank               | 8                                    |
| LoRA alpha              | 16                                   |
| LoRA dropout            | 0.1                                  |
| Target modules          | `q_lin`, `v_lin`                     |
| Maximum sequence length | 128                                  |
| Epochs                  | 2                                    |
| Batch size              | 8                                    |
| Learning rate           | 0.0002                               |
| Weight decay            | 0.01                                 |

## Evaluation

The model achieved a test accuracy of **97.59%** on the balanced evaluation dataset.

Results should be interpreted within the scope of the collected data. Performance may decrease on very short, ambiguous, or previously unseen regional expressions.

## Usage

The adapter must be loaded with its original multilingual DistilBERT base model.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model_name = "distilbert-base-multilingual-cased"
adapter_path = "models/transformer_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

## Limitations

* The model performs binary classification only.
* Oujdi and other Moroccan dialects can share vocabulary and spelling.
* Very short or ambiguous text may produce unreliable predictions.
* The model may reflect limitations or biases present in the training data.
* The current version is intended for research and educational use.

## Authors

Developed by Chaima Menouar and Mohammed Oulhadj under the supervision of Prof. Mohamed Cherradi.

## Data and Usage Terms

The Moroccan Darija portion is derived from the Darija Open Dataset and is subject to the CC BY-NC 4.0 license.

The Oujdi corpus was created by the project team and has not been separately licensed for redistribution. Contact the project authors before redistributing the corpus or using it commercially.

## Project Repository

Full source code, evaluation results, API, interface, and research report:

https://github.com/chaima-menouar/oujdi-dialect-classifier
