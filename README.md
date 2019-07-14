## Tasks

1 - Implement a Language Identification service that returns the language code of the language in which the text is written. The provided data and test will
target Spanish (ES), Portuguese (PT-PT) and English (EN)

2 - Train the system to distinguish between language variants. In this case we wish to distinguish between European Portuguese (PT-PT) and Brazilian Portuguese (PT-BR)

3 - Implement a deep learning model (recommended: a BILSTM tagger) to detect code switching (language mixture) and return both a list of tokens and a list with one language label per token.
To simplify we are going to focus on English and Spanish, so you only need to return for each token either 'en', 'es' or 'other'

## Project description

#### Installation

Before running the scripts and notebooks, install the requirements.txt.

#### Demo scripts

The demo scripts `task1.py`, `task2.py`, and `task3.py` solve the corresponding tasks.
How to run:

```
python task1.py input_file output_file

python task2.py input_file output_file
```

`input_file` should be a text file with one document for a line.

`output_file` will contain labels, one label for a line.

```
python task3.py input_file.tsv output_file
```

`input_file.tsv` should be a .tsv or .csv file in the same format as `train_data.tsv` and `dev_data.tsv` in `code_switching/data`.

`output_file` will contain labels; in one line there will be the labels for one document, separated by commas.

#### Demo notebooks

`demo_notebooks` folder contains notebooks corresponding to the tasks. They demonstrate how to use the models and how to evaluate their quality.


In `code_switching` and `langid` I provided the description of my solutions.

-----------------------------------------------------------------


