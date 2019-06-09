from pathlib import Path
import sys

from langid.language_identification.fasttext_lang_id import FasttextLangId

if __name__ == '__main__':

    if len(sys.argv) < 2:
        test_file = input("Please, enter input file name and output file name:")
    else:
        test_file = sys.argv[1]

    if len(sys.argv) < 3:
        output_file = 'langid_variants_output.txt'
        print(f'Output was saved to {output_file}')
    else:
        output_file = sys.argv[2]


    model = FasttextLangId('langid/models/langid_variants.bin')
    output = model.predict_file(test_file)

    with Path(output_file).open('w') as fout:
        for line in output:
            fout.write(f'{line}\n')
