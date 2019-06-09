from pathlib import Path
import sys

import pandas as pd

from code_switching.code_switching_detection.bilstm_code_switching import BilstmCodeSwitching, load_data, prepare_data

if __name__ == '__main__':

    if len(sys.argv) < 2:
        test_file = input("Please, enter input file name and output file name:")
    else:
        test_file = sys.argv[1]

    if len(sys.argv) < 3:
        output_file = 'code_switching_output.txt'
        print(f'Output was saved to {output_file}')
    else:
        output_file = sys.argv[2]

    model = BilstmCodeSwitching()
    model.load('code_switching/models/model_20ep/')

    data: pd.DataFrame = load_data(test_file)
    prepared_data = prepare_data(data)
    tokenized_texts = [text[0] for text in prepared_data]

    predictions = model.predict(tokenized_texts)

    with Path(output_file).open('w') as fout:
        for line in predictions:
            joined_line = ', '.join(line)
            fout.write(f'{joined_line}\n')
