'''
Script to aggregate UMLS and N2C2 results into a single file.

Each instance comes with a unique ROW_ID, which can be used to match the UMLS and N2C2 results.

UMLS results are stored in JSON format, with each line representing a single instance. 
Full corpus are broken into 20 chunks, each of which is stored in a separate file.

N2C2 results are stored in CSV format, with each line representing a single instance.
Full corpus are broken into 10 chunks, each of which is stored in a separate file.
'''

import os, ast, json
import pandas as pd
from tqdm import tqdm

n2c2_path, umls_path, umls_n2c2_path = 'mimic3_n2c2', 'mimic3_umls', 'mimic3_term'
n2c2_files =  [f for f in os.listdir(n2c2_path) if f.endswith('.csv')]

with open(os.path.join(umls_n2c2_path, 'counts.txt'), 'w') as tf:
    tf.write(','.join(['filname', "file_length", 'no_umls_cnt', 'no_n2c2_cnt', 'no_umls_n2c2_cnt'])+'\n')

for file in n2c2_files:
    print('-'*20, 'Start processing file {}.'.format(file), '-'*20)
    counter = int(file.split('_')[0][5:])
    n2c2_result = pd.read_csv(os.path.join(n2c2_path, file))
    print('Finish loading N2C2 result.')
    n2c2_result.set_index('ROW_ID', inplace=True)
    if counter < 10:
        umls_files = ['chunk'+str(counter*2)+'.json', 'chunk'+str(counter*2 + 1)+'.json']
    else:
        umls_files = ['chunk'+str(counter*2)+'.json']
    
    umls_result = []
    for ufile in umls_files:
        with open(os.path.join(umls_path, ufile)) as uf:
            for line in uf:
                umls_result.append(json.loads(r'{}'.format(line)))
    print('Finish loading UMLS result.')
    
    try:
        assert len(umls_result) == len(n2c2_result)
    except:
        print("File {} has unequal number of instances with {}".format(file, "; ".join(umls_files)))
        break
    
    
    concat_json = []
    concat_df = pd.DataFrame(columns=['ROW_ID', 'text', 'umls_terms', 'umls_terms_indices', 'n2c2_terms', 'n2c2_terms_indices'])
    no_umls_cnt, no_n2c2_cnt,no_umls_n2c2_cnt = 0, 0, 0

    for line in tqdm(umls_result):
        text = line['processed_text']
        new_line  = {'text': text}
        new_row = [line['ROW_ID'], text]

        # make sure that term indices align with the text
        umls_terms, umls_terms_indices = [], []
        line_umls_terms = dict(sorted(line['term_indices'].items(), key=lambda item: item[1][0]))
        for term, indices in line_umls_terms.items():
            umls_terms.append(term)
            umls_terms_indices.append(indices)
        new_line['umls_terms'] = umls_terms
        new_line['umls_terms_indices'] = umls_terms_indices
        new_row.append(umls_terms)
        new_row.append(umls_terms_indices)

        n2c2_terms, n2c2_terms_indices = [], []
        sorted_line_n2c2_result = sorted(ast.literal_eval(n2c2_result.loc[line['ROW_ID'], 'mask_terms']), key=lambda item: item['start'])
        for dic in sorted_line_n2c2_result:
            sidx, eidx = dic['start'], dic['end']
            n2c2_terms.append(text[sidx:eidx])
            n2c2_terms_indices.append([sidx, eidx])
        new_line['n2c2_terms'] = n2c2_terms
        new_line['n2c2_terms_indices'] = n2c2_terms_indices
        new_row.append(n2c2_terms)
        new_row.append(n2c2_terms_indices)

        concat_json.append(new_line)
        concat_df.loc[len(concat_df)] = new_row

        if len(umls_terms) == 0:
            no_umls_cnt += 1
            if len(n2c2_terms) == 0:
                no_umls_n2c2_cnt += 1
                no_n2c2_cnt += 1
        else:
            if len(n2c2_terms) == 0:
                no_n2c2_cnt += 1

                
    
    concat_df.to_csv(os.path.join(umls_n2c2_path, 'chunk'+str(counter)+'_terms.csv'), index=False)
    with open(os.path.join(umls_n2c2_path, 'chunk'+str(counter)+'_terms.json'), 'w') as f:
        for j in concat_json:
            f.write(json.dumps(j)+'\n')
    
    with open(os.path.join(umls_n2c2_path, 'counts.txt'), 'a') as tf:
        tf.write(','.join([file, str(len(concat_json)), str(no_umls_cnt), str(no_n2c2_cnt), str(no_umls_n2c2_cnt)])+'\n')
