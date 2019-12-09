import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as en_stops
from sklearn.neighbors import NearestNeighbors as NN
import pickle
import pysparnn.cluster_index as ci
import numpy as np
from scipy.sparse import csr_matrix
import pdb

f_train_pmids = 'corpus_pubtator_pmids_trng.txt'
f_test_pmids = 'corpus_pubtator_pmids_test.txt'
f_dev_pmids = 'corpus_pubtator_pmids_dev.txt'
f_mm_docs = 'medmentions_context_docs.json'
f_umls_docs = '/scratch/arahimi/cds/umls/umlsdocs.json'
f_test_candidates = '/scratch/arahimi/cds/wiki/wikimentions/IndexWikipedia/mm_test_candidates.txt'
f_mrconso = '/scratch/arahimi/cds/umls/mrconso.json'

def get_mrconso():
    with open(f_mrconso, 'r') as fin:
        return json.load(fin)

def get_pmids(f_pmid):
    pmids = set()
    with open(f_pmid, 'r') as fin:
        for line in fin:
            pmids.add(line.strip())
    return pmids


def getZeshel(f_in, f_out_docs):
    train_pmids = get_pmids(f_train_pmids)
    test_pmids = get_pmids(f_test_pmids)
    dev_pmids = get_pmids(f_dev_pmids)
    train_mentions = []
    test_mentions = []
    dev_mentions = []

    mentions = []

    docs = []
    new_doc = True

    def write_mentions(f_mentions, mentions):
        with open(f_mentions, 'w') as fout:
            for m in mentions:
                fout.write(json.dumps(m))
                fout.write('\n')

    mid = 0
    with open(f_in, 'r') as fin:
        for line in fin:
            if new_doc:
                new_doc = False
                title = ''
                doc = ''
                docid, _, title = line.strip().split('|')
            else:
                if '|' in line and line.strip().split('|')[1] == 'a':
                    fields = line.strip().split('|')
                    doc = '|'.join(fields[2:])
                    doc = title + ' ' + doc
                elif line.strip() != '':
                    _, start_idx, end_idx, text, entity_type, cui = line.strip().split('\t')
                    start_idx, end_idx = int(start_idx), int(end_idx)
                    w_start_idx = doc.count(' ', 0, start_idx)
                    w_end_idx = w_start_idx + text.count(' ')
                    mention = { 'category': 'medmentions', 'label_document_id': cui, 'context_document_id': 'mm_' + docid, 'text': text, 'start_index': w_start_idx, 'end_index': w_end_idx, 'mention_id': 'mm_' + str(mid), "corpus": "medmentions"}
                    if docid in train_pmids:
                        train_mentions.append(mention)
                    elif docid in test_pmids:
                        test_mentions.append(mention)
                    elif docid in dev_pmids:
                        dev_mentions.append(mention)
                    else:
                        raise Exception("docid not in any of test, dev, or train pmids!!", docid)
                    mid += 1
                else:
                    new_doc = True
                    doc_dict = {'document_id': "mm_" + docid, 'text': doc, 'category': 'medmentions'}
                    docs.append(doc_dict)
    write_mentions('train_mentions.json', train_mentions)
    write_mentions('test_mentions.json', test_mentions)
    write_mentions('dev_mentions.json', dev_mentions)
    with open(f_out_docs, 'w') as fout:
        for doc in docs:
            fout.write(json.dumps(doc))
            fout.write('\n')
def get_docids(f_docs, is_mention=False):
    doc_ids = set()
    with open(f_docs, 'r') as fin:
        for line in fin:
            doc_id = json.loads(line.strip())["document_id"] if not is_mention else json.loads(line.strip())["label_document_id"]
            doc_ids.add(doc_id)
    return doc_ids

def get_mention_docs(split="test"):
    mentions = {}
    with open(split + '_mentions.json', 'r') as fin:
        for line in fin:
            mention = json.loads(line.strip())
            mentions[mention["mention_id"]] = mention
    return mentions

def gen_candidates():
    test_mentions = get_mention_docs("test")
    train_mentions = get_mention_docs("train")
    dev_mentions = get_mention_docs("dev")
    mentions = {}
    mentions.update({k: ' '.join(set(v["text"].split()) - en_stops) for k, v in train_mentions.items()})
    mentions.update({k: ' '.join(set(v["text"].split()) - en_stops) for k, v in test_mentions.items()})
    mentions.update({k: ' '.join(set(v["text"].split()) - en_stops) for k, v in dev_mentions.items()})

    mrconso = get_mrconso()
    aliases = {k: " ".join(set(v["alias"]["ENG"]) - en_stops) for k, v in mrconso.items() if "ENG" in v["alias"]}
    mention_ids = sorted(mentions)
    cuis = sorted(aliases)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), max_features=100000)
    print(vectorizer)
    X_cui = vectorizer.fit_transform([aliases[cid] for cid in cuis])
    X_mention = vectorizer.transform([mentions[mid] for mid in mention_ids])
    print(X_cui.shape, X_mention.shape)
    print("indexing...")
    cp = ci.MultiClusterIndex(X_cui, cuis)
    print("searching...")
    ns = cp.search(X_mention, k=64, k_clusters=2, return_distance=False)
    with open('ns.pkl', 'wb') as fout:
        pickle.dump((ns, cuis, mention_ids), fout)
    with open('mm_tfidf_candidatessparsenn.json', 'w') as fout:
        for i, nbrs in enumerate(ns):
            mention_id = mention_ids[i]
            fout.write(json.dumps({"mention_id" : mention_id, "tfidf_candidates": [nbr for nbr in nbrs]}))
            fout.write('\n')


def get_candidates(f_all):
    candidates = {}
    with open(f_all, 'r') as fin:
        for line in fin:
            cand = json.loads(line.strip())
            candidates[cand['mention_id']] = cand['tfidf_candidates']
    return candidates


def compare_candidates(f_all=None, split="test"):
    mrconso = get_mrconso()
    mentions = get_mention_docs(split)
    umls_docs = get_docids(f_umls_docs)
    mm_docs = get_docids(f_mm_docs)
    mention_docs = get_docids('test_mentions.json', is_mention=True)
    print("in mention but not in umlsdocs:", len(mention_docs), len(mention_docs - umls_docs))
    candidate_docs = set()
    hits = 0
    num_mentions = 0
    raw = False
    if not raw and not f_all:
        raise
    if raw:
        with open(f_test_candidates, 'r') as fin:
            for line in fin:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                mention_id, candidates64 = fields
                label = mentions[mention_id]["label_document_id"]
                candidates64 = set(candidates64.split('|||'))
                candidate_docs.update(candidates64)
                if label in candidates64:
                    hits += 1
                num_mentions += 1
    else:
        candidates = get_candidates(f_all)
        for k, v in candidates.items():
            candidate_docs.update(v)
        for m in mentions:
            label = mentions[m]['label_document_id']
            candidates64 = candidates[m]


            if label in candidates64:
                hits += 1
            else:
                try:
                    l_name = mrconso[label]['alias']['ENG'][0]
                    cands =  [mrconso[c]['alias']['ENG'][0] for c in candidates64]
                    print(l_name, cands)
                except Exception as e:
                    print("error", e)
                pdb.set_trace()
            num_mentions += 1
    print("bm25 recall", hits/num_mentions)
    print("in candidates but not in mention docs", len(candidate_docs), len(candidate_docs - mention_docs))
    print("in candidates but not in umls docs", len(candidate_docs), len(candidate_docs - umls_docs))
if __name__ == '__main__':
    #getZeshel('corpus_pubtator.txt', f_mm_docs)
    compare_candidates(f_all='mm_tfidf_candidatessparsenn.json')
    #gen_candidates()




