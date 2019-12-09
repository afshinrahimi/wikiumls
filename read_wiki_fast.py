#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import bz2
import functools
import gensim
import logging
import sys
import re
import mwparserfromhell
import json
import multiprocessing
from collections import defaultdict, Counter
import gzip
import random
import pickle
import pdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as en_stops
from sklearn.neighbors import NearestNeighbors as NN

N_CPU = int(multiprocessing.cpu_count() * 0.8)
logger = logging.getLogger(__name__)
data_path = '../wiki2vec_training/enwiki-latest-pages-articles.xml.bz2'
f_cui_wiki = '../meshdesc_wiki.json'
f_documents = 'related_docs.json.gz'
f_mentions = 'mentions.json'
f_wikilinks = 'wikilinks.txt.gz'
f_valid_wikis = 'all_candidates_lucene18k.txt'
f_cui_mesh = '../../umls/cui_mesh.json'
f_candidates_lucene = './IndexWikipedia/wiki_candidates.txt.gz'
f_candidates_lucene_uniq = './IndexWikipedia/wiki_candidates_perline_uniq.txt.gz'
f_petscan = './pagepile_json_27001.json'
f_pageids = './wikicategories/pageids.json'
f_links_info = 'links_info.pkl'
f_mrconso = '/scratch/arahimi/cds/umls/mrconso.json'
f_snomed_cui = '../../umls/snomed_cui.json'
f_snomed_wiki = './reranking/snomed_wiki.csv'
random.seed(0)


def get_mrconso():
    with open(f_mrconso, 'r') as fin:
        return json.load(fin)

''' class of Wiki Dump Iterator'''
class WikiDumpIter(object):
    def __init__(self, dump_file):
        self.dump_file = dump_file
        self.IGNORE_PAGE = ['wikipedia:', 'category:', 'file:', 'portal:', 'template:', 'mediawiki:',
        'user:', 'help:', 'book:', 'draft:']
    def __iter__(self):
        with bz2.BZ2File(self.dump_file) as f:
            for (title, wiki_text, wiki_page_id) in gensim.corpora.wikicorpus.extract_pages(f):
                if any([title.lower().startswith(namespace) for namespace in self.IGNORE_PAGE]):
                    continue
                #if unicode(title) == "Geordie songwriters’ aliases".decode("utf-8"):
                #    print(wiki_page_id, unicode(title), unicode(wiki_text))
                yield [wiki_page_id, title, wiki_text]
def is_redirect_page(text):
    REDIRECT_REGEXP = re.compile(r"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))", re.IGNORECASE)
    redirect_match = REDIRECT_REGEXP.match(text)
    if not redirect_match:
        return False
    return True
def _wiki_to_raw(text):
    return gensim.corpora.wikicorpus.filter_wiki(text)
def extract_pages(dump_file):
    logger.info('Starting to read dump file...')
    reader = WikiDumpIter(dump_file)
    wiki_pages = []
    pool = multiprocessing.pool.Pool(N_CPU)
    imap_func = functools.partial(pool.imap_unordered, chunksize=100)
    func_name = _process_page
    with gzip.open(f_documents, "wt") as f:
        i = 0
        for page in imap_func(func_name, reader):
            if page is None:
                continue
            f.write(page)
            f.write('\n')
            i += 1
            if i % 1000 == 0:
                logging.info("[write] " + str(i))

def _process_page(page):
    wiki_id, title, text = page
    if is_redirect_page(text):
        return None
    if wiki_id not in valid_titles:
        return None

    text = _wiki_to_raw(text)
    text = ' '.join(text.split()) if text else text
    result = json.dumps({'title': title, 'text': text, 'document_id': wiki_id})



    return result


def extract_pages_addlucene(dump_file):
    logger.info('Starting to read dump file...')
    reader = WikiDumpIter(dump_file)
    wiki_pages = []
    pool = multiprocessing.pool.Pool(N_CPU)
    imap_func = functools.partial(pool.imap_unordered, chunksize=100)
    func_name = _process_page_addlucene
    #now we're appending
    with gzip.open(f_documents, "at") as f:
        i = 0
        for page in imap_func(func_name, reader):
            if page is None:
                continue
            f.write(page)
            f.write('\n')
            i += 1
            if i % 1000 == 0:
                logging.info("[write] " + str(i))

def _process_page_addlucene(page):
    wiki_id, title, text = page
    if is_redirect_page(text):
        return None
    if wiki_id not in docs_remainder:
        return None
    text = _wiki_to_raw(text)
    text = ' '.join(text.split()) if text else text
    result = json.dumps({'title':title, 'text':text, 'document_id':wiki_id})
    return result

def dump_related_documents(wiki_pages, output_file):
    f = gzip.open(output_file, "wt")
    for i, page in enumerate(wiki_pages):
        # one_line = title.encode("utf-8") + "\n"
        #one_line = title.encode("utf-8") + " "
        #one_line += text.encode("utf-8")
        f.write(page)
        if i % 100000 == 0:
            logging.info("[write] " + str(i))
    f.close()

def get_cui_wiki():
     #print('reading cui_wiki: note that titles have underscore instead of spaces here.')
     with open(f_cui_wiki, 'r') as fin:
         cui_wiki = json.load(fin)
     return cui_wiki

def get_wikis():
    wikis = set()
    with open(f_valid_wikis, 'r') as fin:
        for line in fin:
            wikis.add(line.strip().replace('_', ' '))
    return wikis
def get_petscan():
    with open(f_petscan, 'r') as fin:
        pages = json.load(fin)
    petscan_ids = set()
    for title in pages['pages']:
        id = title_id.get(title, None)
        if id:
            petscan_ids.add(id)
        else:
            pass
    return petscan_ids

def get_wikis_candidates_lucene64(f_candidates_lucene):
    candidates = set()
    i = 0
    with gzip.open(f_candidates_lucene, 'rt') as fin:
        for line in fin:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue

            mention, candidates64 = fields

            candidates64 = set(candidates64.split('|||'))
            candidates.update(candidates64)
            if len(candidates) % 100000 == 0:
                print('candidates', len(candidates))
    '''

    with gzip.open(f_candidates_lucene_uniq, 'rt') as fin:
        for line in fin:
            fields = json.loads(line.strip())
            mention, candidates64 = fields['mention_id'], fields['tfidf_candidates']
            candidates.update(set(candidates64))
            i += 1
            if i % 100000 == 0:
                print('read candidates', i)

    '''
    return candidates

def get_related_pages(only_related_wikis=False, topN=None):

     with gzip.open(f_links_info, 'rb') as fin:
         valid_titles, source_target, target_mentions, source_targetmention, p_t_given_m, target_mention_valid, wikis = pickle.load(fin)
         return valid_titles, source_target, target_mentions, source_targetmention, p_t_given_m, target_mention_valid, wikis

     cui_wiki = get_cui_wiki()
     wiki_cui = {v:k for k, v in cui_wiki.items()}
     #remove underscore
     #wikis = set([v.replace('_', ' ') for k, v in cui_wiki.items()])
     #wikis = wikis | get_wikis()
     #wikis = get_wikis_candidates_lucene64(f_candidates_lucene)
     wikis = get_petscan()

     print('#valid/related wiki entities:', len(wikis))
     related_wikis = set()
     source_target = defaultdict(set)
     target_mentions = defaultdict(set)
     source_targetmention = defaultdict(set)
     mention_count = Counter()
     mentiontarget_count = Counter()
     p_t_given_m = defaultdict(int)
     target_mention_valid = defaultdict(set)
     num_links = 0
     with gzip.open(f_wikilinks, 'rt') as fin:
         for line in fin:
             num_links += 1
             if topN and num_links > topN:
                 break
             if num_links % 10000000 == 0:
                 print("links processed: ", num_links)
             fields = line.strip().split('\t')
             if len(fields) != 3:
                 #print(line)
                 continue
             title, mention , target = fields
             target_id = title_id.get(target.replace(' ', '_'), None)
             if target_id in wikis:
                 source_id = title_id.get(title.replace(' ', ' '), None)
                 if not source_id or not target_id:
                     continue
                 mention_count[mention.lower()] += 1
                 mentiontarget_count[(mention.lower(), target_id)] += 1
                 related_wikis.add(source_id)
                 related_wikis.add(target_id)
                 if only_related_wikis:
                     continue
                 source_target[source_id].add(target_id)
                 target_mentions[target_id].add(mention.lower())
                 source_targetmention[source_id].add((target_id, mention))
     for mentiontarget, c in mentiontarget_count.items():
         p_t_given_m[mentiontarget] = mentiontarget_count[mentiontarget] / float(mention_count[mentiontarget[0]])

     if not only_related_wikis:
         for mt, p in p_t_given_m.items():
             if p < 0.7:
                 continue
             m, t = mt
             target_mention_valid[t].add(m)

     with gzip.open(f_links_info, 'wb') as fout:
         pickle.dump((related_wikis, source_target, target_mentions, source_targetmention, p_t_given_m,
                      target_mention_valid, wikis), fout)


     return related_wikis, source_target, target_mentions, source_targetmention, p_t_given_m, target_mention_valid, wikis

def get_document_ids():
    document_ids = set()
    with gzip.open(f_documents, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            docid = obj['document_id']
            document_ids.add(docid)
    return document_ids

class DocDumpIter(object):
    def __init__(self, dump_file):
        self.dump_file = dump_file
    def __iter__(self):
        with gzip.open(self.dump_file, 'rt') as f:
            for line in f:
                yield line
def _dump_mentions(doc):
    doc = json.loads(doc)
    title = doc['title']
    text = doc['text']
    context_document_id = doc['document_id']
    outlinks = source_target[context_document_id]
    outlinks = wikis & outlinks
    if len(outlinks) == 0:
        return None
    inpage_target_mention = defaultdict(set)
    for targetmention in source_targetmention.get(context_document_id, []):
        inpage_target_mention[targetmention[0]].add(targetmention[1])
    mention_id = 0
    for lnk in outlinks:
        label_document_id = lnk
        if not label_document_id:
            continue
        mentions = target_mention_valid[lnk]
        inpage_mentions = inpage_target_mention.get(lnk, {})
        mentions.update(inpage_mentions)

        txtwords = text.split()
        txtwordslower = [w.lower() for w in txtwords]
        all_mentions = []
        for mention in mentions:
            if mention.strip() == '':
                continue
            mwords = mention.split()
            indices = [(i, i+len(mwords)) for i in range(len(txtwordslower)) if txtwordslower[i:i+len(mwords)] == mwords]
            for span in indices:
                d = {'category':0, 'text': ' '.join(txtwords[span[0]: span[1]]), 'end_index': span[1] - 1, 'context_document_id':context_document_id, 'label_document_id':label_document_id, 'mention_id': str(mention_id), 'corpus':'wikipedia', 'start_index':span[0]}
                mention_id += 1
                all_mentions.append(d)
        return all_mentions

def dump_mentions_parallel():
    pool = multiprocessing.Pool(N_CPU)
    mid = 0
    with open(f_mentions, 'w') as fout:
        for mentions in pool.imap_unordered(_dump_mentions, DocDumpIter(f_documents)):
            if mentions:
                for mention in mentions:
                    mention['mention_id'] = str(mid)
                    jstring = json.dumps(mention) + '\n'
                    fout.write(jstring)
                    mid += 1


def get_links(page):
    #handler = WikiXmlHandler()
    #parser = xml.sax.make_parser()
    #parser.setContentHandler(handler)
     # process line
    links = []
    id , page_title, wiki_text = page
    #try:
    #    parser.feed(line)
    #except:
    #    return links
    #if len(handler._pages) == 0 or not handler._pages[0][0]:
    #    return links
    #if any([title.lower().startswith(namespace) for namespace in IGNORE_PAGE]):
    #    continue
    #print(handler._pages[i][0], handler._pages[i][2])
    wiki = mwparserfromhell.parse(wiki_text)

    for lnk in wiki.filter_wikilinks():
        title = lnk.title.strip_code().strip(' ')
        if title.startswith(':'):
            title = title[1:]
        if not title:
            continue
        title = (title[0].upper() + title[1:]).replace('_', ' ')
        if '#' in title:
            title = ' '.join(title.split('#')[:-1])
        if lnk.text:
            text = lnk.text.strip_code()
            # dealing with extended image syntax: https://en.wikipedia.org/wiki/Wikipedia:Extended_image_syntax
            if title.lower().startswith('file:') or title.lower().startswith('image:'):
                text = text.split('|')[-1]
        else:
            text = lnk.title.strip_code()
        if text.strip() == "":
            continue
        a = '\t'.join([page_title, text, title])
        if a[-4:].lower() not in {'.svg', '.png', '.jpg'}:
            links.append(a.encode('utf-8'))

    del wiki
    return links

def extract_links_parallel():
    pool = multiprocessing.Pool(N_CPU)
    with gzip.open(f_wikilinks, 'wt') as fout:
        for lnks in pool.imap_unordered(get_links, WikiDumpIter(data_path)):
            if lnks:
                for lnk in lnks:
                    fout.write(lnk)
                    fout.write('\n')

def downsample(mention_per_entity=10, num_candidates=64):
    all_records = defaultdict(set)
    i = 0
    with open(f_mentions, 'rt') as fin:
        for line in fin:
            i += 1
            if i % 1000000 == 0:
                print('processed mentions to be downsampled:', i)
            mention = json.loads(line.strip())
            target = mention['label_document_id']
            all_records[target].add(line.strip())
    num_all_records = len(all_records)
    print('total number of uniqe entities mentioned:', num_all_records)
    i = 0
    j = 0
    chosen_mention_ids = set()
    chosen_documents = set()
    chosen_context_documents = set()
    chosen_candidates = set()
    with open(f_mentions + '.downsampled', 'wt') as fout:
        for target, mention_set in all_records.items():
            num_mention_set = len(mention_set)
            downsampled_set = random.sample(mention_set, mention_per_entity) if num_mention_set > mention_per_entity else mention_set
            i += num_mention_set
            j += len(downsampled_set)
            for m in downsampled_set:
                m = json.loads(m)
                chosen_mention_ids.add(str(m['mention_id']))
                chosen_context_documents.add(m['context_document_id'])
            chosen_documents.add(m['label_document_id'])
            records = '\n'.join(downsampled_set)
            fout.write(records)
            #if not last item
            if i != num_all_records:
                fout.write('\n')
            if i % 100000 == 0:
                print('writing downsampled entities', i, 'downsampled mentions', j)

    print('writing downsampled entities', i, 'downsampled mentions', j, 'num_chosen_mentions', len(chosen_mention_ids),
            'chosen_docs', len(chosen_documents), 'chosen context docs', len(chosen_context_documents))
    print('downsampling candidates')
    with gzip.open(f_candidates_lucene + '.downsampled.gz', 'wt') as fout:
        with gzip.open(f_candidates_lucene, 'rt') as fin:
            i = 0
            j = 0
            for line in fin:
                i += 1
                if i % 1000000 == 0:
                    print('lucene candidates processed', i, 'written', j)
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    print(line)
                    continue
                mention_id = fields[0]
                if mention_id not in chosen_mention_ids:
                    #print(mention_id, random.sample(chosen_mention_ids, 1))
                    continue
                j += 1
                #get candidates
                candidates = fields[1].split('|||')
                if len(candidates) < num_candidates:
                    print('num bm25 candidates', len(candidates))
                    #choose random candidates, note that is possible a document is chosen more than once with a very very low probability (hope that doesn't happen)
                    extra_candidates = random.sample(chosen_documents, num_candidates - len(candidates))
                    candidates = candidates.extend(extra_candidates)
                record = {"mention_id": mention_id, "tfidf_candidates": candidates}
                chosen_candidates.update(candidates)
                fout.write(json.dumps(record))
                fout.write('\n')
    print('downsampling documents')
    with gzip.open(f_documents + '.downsampled', 'wt') as fout:
        chosen_documents.update(chosen_context_documents)
        chosen_documents.update(chosen_candidates)
        with gzip.open(f_documents, 'rt') as fin:
            i = 0
            j = 0
            for line in fin:
                i += 1
                if i % 100000 == 0:
                    print('documents processed', i, 'downsampled written', j)
                record = json.loads(line.strip())
                document_id = record['document_id']
                #check if docid is in all valid sets
                if document_id not in chosen_documents:
                    j += 1
                    continue
                fout.write(line.strip())
                fout.write('\n')
def split_samples(train=70, val=15, test=15):
    assert train + val + test == 100
    mentions = {}
    candidates = {}
    with open(f_mentions, 'r') as fin:
        for line in fin:
            m = json.loads(line.strip())
            m['mention_id'] = str(m['mention_id'])
            mentions[m['mention_id']] = m
    with gzip.open(f_candidates_lucene, 'rt') as fin:
        for line in fin:
            c = json.loads(line.strip())
            candidates[c['mention_id']] = c
    assert len(mentions) == len(candidates), "the number of mentions and candidates must match but do not"
    ids = [i for i in mentions]
    random.shuffle(ids)
    train_index = len(ids) * 70 // 100
    val_index = len(ids) * 75 //100
    train_ids = ids[0:train_index]
    val_ids = ids[train_index:val_index]
    test_ids = ids[val_index:]

    def write(f_, split, ids, content):
        with open(f_ + split, 'w') as fout:
            for id in ids:
                obj = content[id]
                obj_str = json.dumps(obj)
                fout.write(obj_str)
                fout.write('\n')

    write('./', 'train.json', train_ids, mentions)
    write('./', 'val.json', val_ids, mentions)
    write('./', 'test.json', test_ids, mentions)
    write('./IndexWikipedia/', 'train.json', train_ids, candidates)
    write('./IndexWikipedia/', 'val.json', val_ids, candidates)
    write('./IndexWikipedia/', 'test.json', test_ids, candidates)

def compare_candidates_docs():
      document_ids = get_document_ids()
      badmentions = set()
      i = 0
      j = 0
      indices = []
      mentions = {}

      i = 0
      with open(f_mentions, 'rt') as fin:
          for line in fin:
              i += 1
              if i % 1000000 == 0:
                  print('processed mentions', i)
              mention = json.loads(line.strip())
              mentions[str(mention['mention_id'])] = mention
      num_mentions_read = 0
      num_hit = 0
      num_hit_not_in_docs = 0
      with gzip.open(f_candidates_lucene, 'rt') as fin:
            for line in fin:
                fields = json.loads(line.strip())

                mention, candidates64 = fields['mention_id'], fields['tfidf_candidates']
                label = mentions[mention]['label_document_id']
                candidates = set(candidates64)
                if label in candidates:
                    num_hit += 1
                if label not in document_ids:
                    num_hit_not_in_docs += 1
                num_mentions_read += 1
            print(num_mentions_read, num_hit, num_hit_not_in_docs)

def get_pageids():
    title_id = {}
    '''
    with gzip.open(f_pageids, 'rt') as fin:
        for line in fin:
            line = unicode(line, 'utf-8')
            fields = line.split()
            id = fields[0]
            namespace = fields[1]
            title = fields[2]
            is_redirect = fields[4]
            if namespace != "0" or is_redirect == "1":
                continue
            title_id[title] = id
    '''
    with open(f_pageids, 'r') as fin:
        title_id = json.load(fin)

    '''
    title_id = {}
    with open(f_pageids, 'r') as fin:
        pageids = json.load(fin)
    for page in pageids:
        if (page["namespace"] == u"0") and (page["is_redirect"] == u"0"):
            title_id[page["title"]] = page["id"]
    '''
    return title_id


def recall(f_candidates, f_mentions):
    mentions = {}
    candidates = {}
    with open(f_mentions, 'r') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            mentions[obj["mention_id"]] = obj["label_document_id"]
    with open(f_candidates, 'r') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            candidates[obj["mention_id"]] = set(obj["tfidf_candidates"])
    assert len(mentions) == len(candidates)
    hits = 0
    for m, l in mentions.items():
        if l in candidates[m]:
            hits += 1
    rec = hits / len(mentions)
    print("recall is ", rec)

def get_query_docs(f_candidates, raw=True, convert_raw_to_json=False):
    q_candidates = {}
    if raw:
        with gzip.open(f_candidates, 'rt') as fin:
            for line in fin:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    print(fields)
                    continue
                q, candidates = fields[0], fields[1].split('|||')
                q_candidates[q] = candidates
        if convert_raw_to_json:
            with open(f_candidates + '.json', 'w') as fout:
                for q, c in q_candidates.items():
                    fout.write(json.dumps({'q':q, 'candidates':c}))
                    fout.write('\n')
        return q_candidates
    else:
        with open(f_candidates, 'r') as fin:
            lines = fin.readlines()
        objs = [json.loads(line.strip()) for line in lines]
        q_candidates = {item["q"]: item["candidates"] for item in objs}
        return q_candidates

def query_relevance(f_q_rel, f_page_ids, f_mesh_cui):
    with open(f_mesh_cui, 'r') as fin:
        mesh_cui = json.load(fin)
    with open(f_page_ids, 'r') as fin:
        #has underscore
        title_id = json.load(fin)
    with open(f_q_rel, 'r') as fin:
        mesh_wiki = json.load(fin)
    cui_w = {}
    page_not_found = 0
    cui_not_found = 0
    for m, w in mesh_wiki.items():
        cui = mesh_cui.get(m, None)
        if not cui:
            cui_not_found += 1
        wid = title_id.get(w.replace(' ', '_'), None)
        if wid:
            cui_w[cui] = wid
        else:
            page_not_found += 1
    print(f"page not found: {page_not_found}, cui not found: {cui_not_found}.")
    return cui_w
def recall(query_docs, query_rel, at_k=1000000):
    num_exist = 0
    num_hit = 0.0
    num_q = len(query_docs)
    num_rel = len(query_rel)
    for q in query_docs:
        rel = query_rel.get(q, None)
        if rel:
            num_exist += 1
            docs = query_docs[q][:at_k]
            if rel in docs:
                num_hit += 1
    print(f"num queries: {num_q}, num relevants: {num_rel}, num queries in relevants: {num_exist}, recall at {at_k}: {num_hit/num_exist}")
    return num_hit/num_exist


def create_neural_ranking_data(q_r, q_docs, f_wiki_aliases, f_cui_aliases, f_page_ids, big_test=False, all_train=False, entest=False):
    #if entest is true, only the concept name is used as query
    with open(f_page_ids, 'r') as fin:
        #has underscore
        title_id = json.load(fin)
    with gzip.open(f_cui_aliases, 'rt') as fin:
        cui_alias = json.load(fin)
    wiki_alias = {}
    with gzip.open(f_wiki_aliases, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = title_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = obj['aliases']

    if not big_test:
        query_ids = [q for q in q_r]
        random.shuffle(query_ids)
        assert len(query_ids) > 17000
        if not all_train:
            train_qs = query_ids[0:10000]
            test_qs = query_ids[10000:15000]
            dev_qs = query_ids[15000:]
        else:
            train_qs = query_ids
        def write_instances(split, qs):
            label_string = 'index' if split== 'test' else 'Quality'
            with open(f"reranking_entest/{split}.tsv", 'w') as fout:
                fout.write(f'{label_string}\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                for i, qid in enumerate(qs):
                    if qid not in cui_alias:
                        continue
                    if entest:
                        string1 = cui_alias[qid]['name']
                        if not string1:
                            continue
                    else:
                        string1 = ' , '.join(cui_alias[qid]['aliases'])
                    qdocs = q_docs.get(qid, None)
                    if not qdocs:
                        continue
                    for doc in qdocs:
                            label = "1" if q_r[qid] == doc else "0"
                            if split == 'test':
                                label = qid + "_" + doc
                            string2 = wiki_alias.get(doc, None)
                            if not string2:
                                continue
                            string2 = ' , '.join(string2)
                            instance = '\t'.join([label, qid, doc, string1, string2])
                            fout.write(instance)
                            fout.write('\n')
        if not all_train:
            write_instances('train', train_qs)
            write_instances('test', test_qs)
            write_instances('dev', dev_qs)
        else:
            write_instances('trainbig', train_qs)

    elif not all_train:
        def write_instances(split, qs):
            label_string = 'index' if 'test' in split else 'Quality'
            with open(f"reranking/{split}.tsv", 'w') as fout:
                fout.write(f'{label_string}\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                for i, qid in enumerate(qs):
                    if qid not in cui_alias:
                        continue
                    string1 = ' , '.join(cui_alias[qid]['aliases'])
                    qdocs = q_docs.get(qid, None)
                    if not qdocs:
                        continue
                    for doc in qdocs:
                        if not 'test' in split:
                            label = "1" if q_r[qid] == doc else "0"
                        else:
                            label = qid + "_" + doc
                        string2 = wiki_alias.get(doc, None)
                        if not string2:
                            continue
                        string2 = ' , '.join(string2)
                        instance = '\t'.join([label, qid, doc, string1, string2])
                        fout.write(instance)
                        fout.write('\n')
        write_instances('big_test', sorted([q for q in q_docs]))

def create_neural_ranking_data_englishtest(q_r, q_docs, f_wiki_aliases, f_cui_aliases, f_page_ids, big_test=False, all_train=False):
    with open('./reranking/test_cuis.txt', 'r') as fin:
        lines = fin.readlines()
    test_cuis = [l.strip() for l in lines]
    with open(f_page_ids, 'r') as fin:
        #has underscore
        title_id = json.load(fin)
    with gzip.open(f_cui_aliases, 'rt') as fin:
        cui_alias = json.load(fin)
    wiki_alias = {}
    with gzip.open(f_wiki_aliases, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = title_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = obj['aliases']

    if not big_test:
        query_ids = [q for q in q_r]
        random.shuffle(query_ids)
        assert len(query_ids) > 17000
        if not all_train:
            train_qs = query_ids[0:10000]
            test_qs = query_ids[10000:15000]
            dev_qs = query_ids[15000:]
        else:
            train_qs = query_ids
        def write_instances(split, qs):
            label_string = 'index' if split== 'test' else 'Quality'
            with open(f"reranking/{split}.tsv", 'w') as fout:
                fout.write(f'{label_string}\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                for i, qid in enumerate(qs):
                    if qid not in cui_alias:
                        continue
                    string1 = ' , '.join(cui_alias[qid]['aliases'])
                    qdocs = q_docs.get(qid, None)
                    if not qdocs:
                        continue
                    for doc in qdocs:
                            label = "1" if q_r[qid] == doc else "0"
                            if split == 'test':
                                label = qid + "_" + doc
                            string2 = wiki_alias.get(doc, None)
                            if not string2:
                                continue
                            string2 = ' , '.join(string2)
                            instance = '\t'.join([label, qid, doc, string1, string2])
                            fout.write(instance)
                            fout.write('\n')
        if not all_train:
            write_instances('train', train_qs)
            write_instances('test', test_qs)
            write_instances('dev', dev_qs)
        else:
            write_instances('trainbig', train_qs)

    elif not all_train:
        def write_instances(split, qs):
            label_string = 'index' if 'test' in split else 'Quality'
            with open(f"reranking/{split}.tsv", 'w') as fout:
                fout.write(f'{label_string}\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                for i, qid in enumerate(qs):
                    if qid not in cui_alias:
                        continue
                    string1 = ' , '.join(cui_alias[qid]['aliases'])
                    qdocs = q_docs.get(qid, None)
                    if not qdocs:
                        continue
                    for doc in qdocs:
                        if not 'test' in split:
                            label = "1" if q_r[qid] == doc else "0"
                        else:
                            label = qid + "_" + doc
                        string2 = wiki_alias.get(doc, None)
                        if not string2:
                            continue
                        string2 = ' , '.join(string2)
                        instance = '\t'.join([label, qid, doc, string1, string2])
                        fout.write(instance)
                        fout.write('\n')
        write_instances('big_test', sorted([q for q in q_docs]))

def big_test():
    q_docs = get_query_docs('IndexWikipedia/umls2wiki_candidates.json', raw=False)

    create_neural_ranking_data(None, q_docs, '../wiki_aliases.json.gz', '../../umls/cui_alias.json.gz', './wikicategories/pageids.json', big_test=True)

def get_rerank_qdocs(f_test, f_predictions, partial=False):
    relevance = {}
    with open(f_test, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                pass
            else:
                fields = line.strip().split('\t')
                cui, docid = fields[0].split('_')
                relevance[i-1] = [cui, docid, 0]
    with open(f_predictions, 'r') as fin:
        for i, line in enumerate(fin):
            if partial and len(line.strip().split()) != 2:
                break
            rel = line.strip().split('\t')[1]
            rel = float(rel)
            relevance[i][2] = rel
    cui_docscore = {}
    for k, v in relevance.items():
        cui = v[0]
        docid = v[1]
        score = v[2]
        docscore = cui_docscore.get(cui, [[], []])
        docscore[0].append(docid)
        docscore[1].append(score)
        cui_docscore[cui] = docscore
    q_docs = {}
    for cui, docscore in cui_docscore.items():
        rank = [docid for _, docid in sorted(zip(docscore[1], docscore[0]), reverse=True)]
        q_docs[cui] = rank
    return q_docs



def rerank_recall(q_r, q_docs=None):
    exp_recalls = {}
    for exp in ['en', 'multi', 'biobert', 'multientest', 'zeroshot', '0shot']:
        print('experiment', exp)
        q_docs = get_rerank_qdocs('./reranking/test.tsv' if exp!='0shot' else './reranking/test_0shot.tsv', f'./reranking/test_results_small_{exp}.tsv')
        recalls = []
        for at_k in range(1, 65):
            r = recall(q_docs, q_r, at_k=at_k)
            recalls.append(r)
        exp_recalls[exp] = recalls
        #plt.plot(range(1, 65), recalls)
        #plt.savefig(f'rerank_recall_{exp}.pdf')
        #plt.close()

    return exp_recalls

def bm25_recall(q_r, q_docs=None):
    if not q_docs:
        q_docs = get_query_docs('IndexWikipedia/snmmsh2wiki_candidates.json', raw=False)
    recalls = []
    for at_k in range(1, 65):
        r = recall(q_docs, q_r, at_k=at_k)
        recalls.append(r)
    #plt.plot(range(1, 65), recalls)
    #plt.savefig('bm25_recall.pdf')
    #plt.close()
    return recalls

def get_recall(q_r, q_docs, k=64):
    recalls = []
    for at_k in range(1, k+1):
        r = recall(q_docs, q_r, at_k=at_k)
        recalls.append(r)
    return recalls

def total_recall():
    q_r = query_relevance('../meshdesc_wiki.json', './wikicategories/pageids.json',  '../../umls/mesh_cui.json')
    rerank_recalls = rerank_recall(q_r)
    bm25_recalls = bm25_recall(q_r)
    charbm25_docs = char_bm25('charbm25_tfidf_candidates.json')
    char_recalls = bm25_recall(q_r, q_docs=charbm25_docs)
    charrevbm25_docs = char_bm25('charbm25_tfidfrev_candidates.json')
    charrev_recalls = bm25_recall(q_r, q_docs=charrevbm25_docs)

    def set_size(width, fraction=1):
        """ Set aesthetic figure dimensions to avoid scaling in latex.
        Parameters
        ----------
        width: float
                Width in pts
        fraction: float
                Fraction of the width which you wish the figure to occupy
        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
        """
        # Width of figure
        fig_width_pt = width * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        # Golden ratio to set aesthetic figure height
        golden_ratio = (5**.5 - 1) / 2
        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio
        fig_dim = (fig_width_in, fig_height_in)
        return fig_dim
    width = 455.24408
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))

    rerank_recalls.update({'BM25': bm25_recalls})
    rerank_recalls.update({'BM25Char': char_recalls})
    rerank_recalls.update({'BM25CharRev':charrev_recalls})
    for exp, recalls in rerank_recalls.items():
        print(exp)
        print_tikz_plot(exp, recalls)
        if exp == "en":
            label = "BERT-EN"
        elif exp == "multi":
            label = "BERT-Multi"
        elif exp == "BM25":
            label = "BM25 (Word)"
        elif exp == "BM25Char":
            label = "BM25 (Char)"
        elif exp == "biobert":
            label = "BioBERT"
        elif exp == 'BM25CharRev':
            label = 'BM25 (Char Rev)'
        elif exp == "multientest":
            label = "multientest"
        elif exp == "zeroshot":
            label = "zeroshot"
        elif exp == "0shot":
            label = "0shot"
        ax.plot(range(1, 65), recalls, label=label)
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Recall')
    #plt.title("Recall at K for retrieving a Wikipedia page for a UMLS concept using BM25, BERT-Large-English, and BERT-Multilingual. UMLS concept aliases in multiple languages are used as query, and Wikipedia titles, and their aliases in multiple languages in Wikidata are used as documents.")
    #plt.title()
    plt.savefig('total_recall.pdf', bbox_inches='tight')
    plt.close()


def char_bm25(f_qdocs=None):
    if f_qdocs:
        q_docs = {}
        with open(f_qdocs, 'r') as fin:
            objs = [json.loads(line.strip()) for line in fin.readlines()]
        q_docs = {obj['q']:obj['candidates'] for obj in objs}
        return q_docs


    f_wiki_aliases = '../wiki_aliases.json.gz'
    f_page_ids =  './wikicategories/pageids.json'
    q_r = query_relevance('../meshdesc_wiki.json', './wikicategories/pageids.json',  '../../umls/mesh_cui.json')
    test_cuis = set()
    with open('./reranking/test.tsv', 'r') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            cui = line.split('\t')[0].split('_')[0]
            test_cuis.add(cui)
    mrconso = get_mrconso()
    mrconso = {q:v for q, v in mrconso.items() if q in test_cuis}
    cui_alias = {}
    for q, v in mrconso.items():
        if 'alias' not in v:
            continue
        names = set()
        for lang in v['alias']:
            names.update(set(' '.join(v['alias'][lang]).split()))
        names = names - en_stops
        cui_alias[q] = ' '.join(names)
    with open(f_page_ids, 'r') as fin:
        #has underscore
        title_id = json.load(fin)
    wiki_alias = {}
    with gzip.open(f_wiki_aliases, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = title_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = ' '.join(set(' '.join(obj['aliases']).split()) - en_stops)
    cuis = sorted(test_cuis)
    wikis = sorted(wiki_alias)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), max_features=100000)
    print(vectorizer)
    X_cui = vectorizer.fit_transform([cui_alias[mid] for mid in cuis])
    X_wiki = vectorizer.transform([wiki_alias[cid] for cid in wikis])
    print(X_wiki.shape, X_cui.shape)

    nbrs = NN(n_neighbors=64, algorithm='brute', metric='cosine', leaf_size=64, n_jobs=10)
    print("fitting nn...")
    nbrs.fit(X_wiki)
    print("finding nbrs...")
    ns = nbrs.kneighbors(X_cui, return_distance=False)
    with open('ns_bruterev.pkl', 'wb') as fout:
        pickle.dump((ns, wikis, cuis), fout)
    I = ns
    i = 0
    j = 0
    q_docs = {}
    with open('charbm25_tfidfrev_candidates.json', 'w') as fout:
        for i in range(I.shape[0]):
            cui = cuis[i]
            nbrs = []
            for j in range(I.shape[1]):
                nbr = I[i, j]
                nbrs.append(wikis[nbr])
            fout.write(json.dumps({"q" : cui, "candidates": nbrs}))
            fout.write('\n')
            q_docs[q] = nbrs
    return q_docs

def print_tikz_plot(experiment, recalls):
    for i in [1, 2, 4, 8, 16, 32, 64]:
        print(f"({i}, {recalls[i-1]:.2f})", end=" ")
        #if (i+1) % 8 == 0:
        #    print()
    print()

def get_rankings(f_results='./results.json'):
    q_r = query_relevance('../meshdesc_wiki.json', './wikicategories/pageids.json',  '../../umls/mesh_cui.json')
    experiments = {}
    wordbm25_docs = get_query_docs('IndexWikipedia/snmmsh2wiki_candidates.json', raw=False)
    experiments['WBM25'] = [wordbm25_docs, get_recall(q_r, wordbm25_docs)]
    charbm25_docs = char_bm25('charbm25_tfidf_candidates.json')
    experiments['CBM25'] = [charbm25_docs, get_recall(q_r, charbm25_docs)]
    charbm25rev_docs = char_bm25('charbm25_tfidfrev_candidates.json')
    experiments['CRBM25'] = [charbm25rev_docs, get_recall(q_r, charbm25rev_docs)]
    q_docs_en = get_rerank_qdocs('./reranking/test.tsv', f'./reranking/test_results_small_en.tsv')

    experiments['en'] = [q_docs_en, get_recall(q_r, q_docs_en)]
    q_docs_multi = get_rerank_qdocs('./reranking/test.tsv', f'./reranking/test_results_small_multi.tsv')
    experiments['multi'] = [q_docs_multi, get_recall(q_r, q_docs_multi)]
    q_docs_biobert = get_rerank_qdocs('./reranking/test.tsv', f'./reranking/test_results_small_biobert.tsv')
    experiments['biobert'] = [q_docs_biobert, get_recall(q_r, q_docs_biobert)]
    q_docs_zeroshot = get_rerank_qdocs('./reranking/test.tsv', f'./reranking/test_results_small_zeroshot.tsv')
    experiments['zeroshot'] = [q_docs_zeroshot, get_recall(q_r, q_docs_zeroshot)]
    q_docs_multientest = get_rerank_qdocs('./reranking/test.tsv', f'./reranking/test_results_small_multientest.tsv')
    experiments['multientest'] = [q_docs_multientest, get_recall(q_r, q_docs_multientest)]

    with open(f_results, 'w') as fout:
        json.dump(experiments, fout)

def get_dataset_statistics():
    mrconso = get_mrconso()
    q_r = query_relevance('../meshdesc_wiki.json', './wikicategories/pageids.json',  '../../umls/mesh_cui.json')
    title_id = get_pageids()
    wiki_alias = {}
    with gzip.open('../wiki_aliases.json.gz', 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = title_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = obj['aliases']
    print(f"#aligned:{len(q_r)}")
    print(f"#mrconso:{len(mrconso)}")
    num_langs = 0
    num_alias = 0
    num_langs_mesh = 0
    num_alias_mesh = 0
    total_mesh = 0
    total_mrconso = 0
    for cui, val in mrconso.items():
        total_mrconso += 1
        num_langs += len(val['alias']) if 'alias' in val else 0
        if cui in q_r:
            total_mesh += 1
            num_langs_mesh += len(val['alias']) if 'alias' in val else 0

        if 'alias' in val:
            for lang, names in val['alias'].items():
                num_alias += len(names)
                if cui in q_r:
                    num_alias_mesh += len(names)

    print(f"mshlan {num_langs_mesh} mshalias {num_alias_mesh} mshtotal {total_mesh} mshlangper {num_langs_mesh/total_mesh} mshaliasper {num_alias_mesh/total_mesh} consolan {num_langs} consoalias {num_alias} consototal {total_mrconso} consolangper {num_langs/total_mrconso} consoaliasper {num_alias/total_mrconso}")

    num_wiki = 0
    num_alias = 0
    for pid, alias in wiki_alias.items():
        num_wiki += 1
        num_alias += len(alias.split(','))
    print(f" num_alias {Num_alias} {num_alias/num_wiki}")


def create_neural_ranking_data_wiki2wiki(f_candidates, f_wiki_aliases_lang, f_page_ids, num_english=2):
    q_docs = get_query_docs(f_candidates, raw=False, convert_raw_to_json=False)
    q_docs = {q:docs for q, docs in q_docs.items() if q in docs[0:10] and len(docs)>1}
    with open(f_page_ids, 'r') as fin:
        #has underscore
        title_id = json.load(fin)
    id_title = {v:k for k,v in title_id.items()}

    wiki_alias = {}
    with gzip.open(f_wiki_aliases_lang, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = title_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = obj['aliases']
    qs = [q for q in q_docs]
    random.shuffle(qs)
    num_train = int(0.8 * len(qs))
    train_qs = qs[0:num_train]
    dev_qs = qs[num_train:]
    def write_instances(split, qs):
        label_string = 'index' if split== 'test' else 'Quality'
        with open(f"reranking/{split}_w2w.tsv", 'w') as fout:
            #fout.write(f'{label_string}\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
            for i, qid in enumerate(qs):
                if qid not in wiki_alias:
                    continue
                aliases = wiki_alias[qid]
                first_aliases = []
                second_aliases = set()
                for lang in aliases:
                    if lang == 'en':
                        num_first = 1
                        first_aliases = aliases[lang][0:num_first]
                        second_aliases.update(set(aliases[lang][num_first:]))
                    else:
                        second_aliases.update(set(aliases[lang]))

                string1 = ' , '.join(first_aliases)
                qdocs = q_docs.get(qid, None)
                if not qdocs:
                    continue
                for doc in qdocs[0:10]:
                        aliases_negative = set()
                        label = "1" if qid == doc else "0"
                        if split == 'test':
                            label = qid + "_" + doc
                        if label == "1":
                            string2 = ' , '.join(second_aliases)
                        else:
                            if doc not in wiki_alias:
                                continue
                            aliases = wiki_alias[doc]
                            for lang in aliases:
                                aliases_negative.update(set(aliases[lang]))
                            if len(aliases_negative) == 0:
                                continue
                            string2 = ' , '.join(aliases_negative)
                        if string1.strip() == '' or string2.strip() == '':
                            continue
                        instance = '\t'.join([label, qid, doc, string1, string2])
                        fout.write(instance)
                        fout.write('\n')
    write_instances('train', train_qs)
    write_instances('dev', dev_qs)

def save_qdocs(f_, q_docs):
    with open(f_, 'w') as fout:
        json.dump(q_docs, fout)
def get_qdocs(f_):
    with open(f_, 'r') as fin:
        return json.load(fin)

def get_snomed_manual(f_snomed_wiki, f_snomed_cui, f_cui_aliases, f_wiki_aliases):
    mrconso = get_mrconso()
    meshdesc_wiki = get_cui_wiki()
    with open('../../umls/mesh_cui.json', 'r') as fin:
        mesh_cui = json.load(fin)
    alignedcui_wiki = {mesh_cui[msh]:wiki for msh, wiki in meshdesc_wiki.items() if msh in mesh_cui}
    with gzip.open(f_cui_aliases, 'rt') as fin:
        cui_alias = json.load(fin)

    with open(f_snomed_wiki, 'r') as fin:
        lines = fin.readlines()
    snomed_wikititle = {line.split(',')[0].strip():line.split(',')[1].strip() for line in lines[1:] if '#' not in line}
    with open(f_snomed_cui, 'r') as fin:
        snomed_cui = json.load(fin)
    page_id = get_pageids()
    id_page = {v:k for k, v in page_id.items()}
    wiki_alias = {}
    with gzip.open(f_wiki_aliases, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = page_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = obj['aliases']

    cui_wiki = {}
    for sid, wtitle in snomed_wikititle.items():
        cui = snomed_cui.get(sid, None)
        if not cui:
            continue
        pid = page_id.get(wtitle, None)
        if not pid:
            continue
        cui_wiki[cui] = pid

    cui_wiki = {k:v for k, v in cui_wiki.items() if k not in alignedcui_wiki}
    manual_lines = []
    with open('reranking/big_test.tsv', 'r') as fin:
        for line in fin:
            cui = line.split('\t')[0].split('_')[0]
            if cui in cui_wiki:
                manual_lines.append(line)
    with open('reranking/manual_test.tsv', 'w') as fout:
        for line in manual_lines:
            fout.write(line.strip())
            fout.write('\n')
    print(f"num eval snomed wiki: {len(cui_wiki)}")

    q_docs = get_query_docs('IndexWikipedia/snmmsh2wiki_candidates.json', raw=False)

    print(bm25_recall(cui_wiki, q_docs))
    #q_docs = get_rerank_qdocs('./reranking/big_test.tsv', f'./reranking/test_results_small_multibig.tsv', partial=True)
    #save_qdocs('reranking/q_docs_multibig.json', q_docs)
    q_docs_multi = get_qdocs('reranking/q_docs_multibig.json')
    print(bm25_recall(cui_wiki, q_docs_multi))
    n = 1
    for cui, wiki in cui_wiki.items():
        if cui in q_docs and wiki in q_docs[cui][0:n] and wiki not in q_docs_multi[cui][0:n]:
            print(json.dumps({"cui":cui, "wiki": page_id.get(wiki, wiki), 'umls': cui_alias[cui], 'wiki': wiki_alias[wiki], 'bm': [id_page[id] for id in q_docs[cui][0:n+2]], 'multi': [id_page[id] for id in q_docs_multi[cui][0:n+2]]}))
    num_langs = 0
    num_alias = 0
    num_total = 0
    for cui in cui_wiki:
        if cui not in mrconso:
            continue
        num_total += 1
        alias = mrconso[cui]['aliases']
        num_langs += len(alias)
        for l, v in alias.items():
            num_alias += len(v)
            print(l, len(v))
    print(f"{num_total} {num_langs/num_total} {num_alias/num_total}")


    return cui_wiki
def create_train_zeroshot(f_out_train):
    f_wiki_aliases_lang='../wiki_aliases_langs.json.gz'
    q_docs = get_query_docs('IndexWikipedia/snmmsh2wiki_candidates.json', raw=False)
    q_docs = {q:docs[0:3] for q, docs in q_docs.items() if len(docs)>5}
    page_id = get_pageids()
    id_page = {v:k for k, v in page_id.items()}
    wiki_alias = {}
    with gzip.open(f_wiki_aliases_lang, 'rt') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            #has underscore
            title = obj['url']
            pid = page_id.get(title, None)
            if not pid:
                continue
            wiki_alias[pid] = obj['aliases']
    with open(f_out_train, 'w') as fout:
        #select 2 english and 1 other
        for q, docs in q_docs.items():
            #ignore q
            aliases = {d: wiki_alias[d] for d in docs if d in wiki_alias}
            for d, alias in aliases.items():
                if 'en' not in alias:
                    continue
                alias['en'] = list(set(alias['en']))
                first = alias['en'][0:2]
                second = alias['en'][2:]
                langs = [l for l in alias if l != 'en']
                chosen_lang = 'en'
                if len(langs) > 0:
                    chosen_lang = random.sample(langs, 1)[0]
                    first.append(alias[chosen_lang][0])
                    second.extend(alias[chosen_lang][1:])
                for l in alias:
                    if l != 'en' and l != chosen_lang:
                        second.extend(alias[l])
                second = set(second)
                first = set(first)
                second = second - first if len(second) > 1 else second
                if len(first) == 0 or len(second) == 0:
                    continue
                second = list(second)
                fout.write(f"1\t{d}\t{d}\t{' , '.join(list(first))}\t{' , '.join(second)}\n")
                for doc2, alias in aliases.items():
                    if doc2 == d:
                        continue
                    second = []
                    for l, names in alias.items():
                        second.extend(names)
                    second = set(second)
                    second = second - first if len(second) > 1 else second
                    if len(second) == 0:
                        continue
                    second = list(set(second))
                    fout.write(f"0\t{d}\t{doc2}\t{' , '.join(first)}\t{' , '.join(second)}\n")








if __name__ == '__main__':
    #create_train_zeroshot(f_out_train='reranking/train_w2wgood.tsv')
    #sys.exit()
    get_snomed_manual(f_snomed_wiki, f_snomed_cui, '../../umls/cui_alias.json.gz', '../wiki_aliases.json.gz' )
    sys.exit(0)
    #create_neural_ranking_data_wiki2wiki(f_candidates='IndexWikipedia/wiki2wiki_candidates.json',
    #        f_wiki_aliases_lang='../wiki_aliases_langs.json.gz',
    #        f_page_ids='./wikicategories/pageids.json', num_english=1)
    #get_dataset_statistics()
    #sys.exit()
    #get_rankings()
    #q_docs = char_bm25()
    #q_r = query_relevance('../meshdesc_wiki.json', './wikicategories/pageids.json',  '../../umls/mesh_cui.json')
    #bm25_recall(q_r, q_docs)
    #sys.exit()
    total_recall()
    sys.exit(0)
    #big_test()
    #sys.exit(0)
    q_r = query_relevance('../meshdesc_wiki.json', './wikicategories/pageids.json',  '../../umls/mesh_cui.json')
    print(f"{len(q_r)} query relevance found.")

    q_docs = get_query_docs('IndexWikipedia/snmmsh2wiki_candidates.json', raw=False)
    create_neural_ranking_data(q_r, q_docs, '../wiki_aliases.json.gz', '../../umls/cui_alias.json.gz', './wikicategories/pageids.json', all_train=False, entest=True)
    #recall("IndexWikipedia/train.json", "train.json")
    sys.exit(0)
    #title_id = get_pageids()
    #compare_candidates_docs()
    #sys.exit()
    #downsample()
    #split_samples()
    sys.exit()
    #print("extracting links from wikipedia...")
    #extract_links_parallel()
    add_extra_docs_from_candidates = True
    document_ids = get_document_ids()
    candidate_doc_ids = get_wikis_candidates_lucene64(f_candidates_lucene)
    docs_remainder = candidate_doc_ids - document_ids
    print("writing extra candidate documents to", f_documents, len(docs_remainder))
    extract_pages_addlucene(dump_file=data_path)
    document_ids = get_document_ids()
    docs_remainder = candidate_doc_ids - document_ids
    print("writing extra candidate documents to", f_documents, len(docs_remainder))
    sys.exit()
    #pdb.set_trace()
    #print("hello")
    #print("getting related pages from petscan and p(target|mention)")
    valid_titles, source_target, target_mentions, source_targetmention, p_t_given_m, target_mention_valid, wikis = get_related_pages(only_related_wikis=add_extra_docs_from_candidates, topN=None)


    #sys.exit()

    #print("reading candidates from 64 candidates per mention...")
    #lucene candidates that are not already written to docs
    #print("extracting docs...")
    #extract_pages(dump_file=data_path)
    #only run this when you have created the mention-64candidates file

    #print("extracting mentions...")
    #dump_mentions_parallel()


