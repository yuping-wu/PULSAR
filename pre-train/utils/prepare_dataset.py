import os, random, json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

random.seed(42)

if __name__ == '__main__':
    path = 'mimic3_term'
    new_path = 'Datasets/Train'
    files = [f for f in os.listdir(path) if f.endswith('json')]
    additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
    for file in files:
        dataset = []
        print('Loading file: ', file)
        with open(os.path.join(path, file)) as f:
            for line in f:
                dataset.append(json.loads(r'{}'.format(line)))
        print('Number of data in the file {} is {}.'.format(file, len(dataset)))

        new_dataset = []
        # process data: decide whether to mask UMLS/N2C2 terms or sentences
        for _, item in tqdm(enumerate(dataset)):
            text, umls_indices, n2c2_indices = item['text'], item['umls_terms_indices'], item['n2c2_terms_indices']
            mask_sent = False
            if umls_indices or n2c2_indices:
                if umls_indices and n2c2_indices:
                    choice = random.uniform(0, 1)
                    if choice < 0.7:
                        terms = umls_indices
                    else:
                        terms = n2c2_indices
                elif umls_indices:
                    terms = umls_indices
                else:
                    terms = n2c2_indices
            # terms are list of list (begin/end index)
            else:
                mask_sent = True
                sents = sent_tokenize(text)
                # if there is no terms to mask, randomly mask sentence
                terms = sorted(random.sample(range(0, len(sents)), k=int(len(sents)*0.15)))
            # terms are list of integers (sentence index)
        
            # praper source and target
            labels = []

            if not mask_sent:
                ssidx = 0
                text_spans = []
                for sidx, eidx in terms:
                    text_spans += [text[ssidx:sidx], text[sidx:eidx]]
                    ssidx = eidx
                text_spans.append(text[eidx:])
            else:
                text_spans = sents
            
            for idx, indices in enumerate(terms):
                choice = random.uniform(0, 1)
                jdx = min(idx, len(additional_special_tokens) - 1)
                if not mask_sent:
                    to_mask_idx = idx*2 + 1
                    mask_span = text[indices[0]:indices[1]]
                    assert mask_span == text_spans[to_mask_idx]
                    labels += [additional_special_tokens[jdx], mask_span]
                    if choice < 0.8:
                        # 80% of the time, we replace masked term/sentence with tokenizer.mask_token ([MASK])
                        # this mask token is specific to tokenizer
                        text_spans[to_mask_idx] = additional_special_tokens[jdx]
                    elif choice < 0.9:
                        # 10% of the time, we replace masked term/sentence with random term/sentence
                        random_idx = random.choice(terms)
                        text_spans[to_mask_idx] = text[random_idx[0]:random_idx[1]]
                    else:
                        # 10% of the time, we keep the masked term/sentence unchanged
                        pass
                else:
                    labels += [additional_special_tokens[jdx], sents[indices]]
                    if choice < 0.8:
                        text_spans[indices] = additional_special_tokens[jdx]
                    elif choice < 0.9:
                        random_idx = random.choice(len(sents))
                        text_spans[indices] = sents[random_idx]
                    else:
                        pass
            
            if mask_sent:
                new_text = ' '.join(text_spans)
            else:
                new_text = ''.join([t for t in text_spans if t])
            jdx = min(len(terms), len(additional_special_tokens) - 1)
            target = " ".join(labels+[additional_special_tokens[jdx]])
            
            new_dataset.append({'text': new_text, 'target': target})
        
        # save new_dataset
        with open(os.path.join(new_path, file), 'w') as outf:
            for nd in new_dataset:
                outf.write(json.dumps(nd))
                outf.write('\n')
