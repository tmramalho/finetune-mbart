from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

en_data = []
jp_data = []

with open('WikiMatrix.en-ja.tsv') as fp:
    for line in tqdm(fp, total=3895992):
        line_data = line.rstrip().split('\t')
        en_data.append(line_data[1] + '\n')
        jp_data.append(line_data[2] + '\n')

with open('en-ja/en-ja.bicleaner05.txt') as fp:
    for line in tqdm(fp, total=10120013):
        line_data = line.rstrip().split('\t')
        en_data.append(line_data[2] + '\n')
        jp_data.append(line_data[3] + '\n')

total_test = 60000
en_train, en_subtotal, jp_train, jp_subtotal = train_test_split(
        en_data, jp_data, test_size=total_test, random_state=42)

en_test, en_val, jp_test, jp_val = train_test_split(
        en_subtotal, jp_subtotal, test_size=0.5, random_state=42)

file_mapping = {
    'train.en_XX': en_train,
    'train.ja_XX': jp_train,
    'valid.en_XX': en_val,
    'valid.ja_XX': jp_val,
    'test.en_XX': en_test,
    'test.ja_XX': jp_test,

}
for k, v in file_mapping.items():
    with open(f'preprocessed/{k}', 'w') as fp:
        fp.writelines(v)
