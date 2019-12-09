import json
import gzip
import pdb
import multiprocessing as mp

infile = 'meshed.json.gz'
outfile = 'meshdesc_wiki.json'
ncpu = mp.cpu_count() - 10
codes = {'meshcode':'P672', 'meshdesc':'P486', 'icd10':'P4229', 'umls':'P2892'}
code_counts = {k:0 for k in codes}
cui_wiki = {}
extraction_code = 'meshdesc'

def wikidata_iter(filename):
    with gzip.open(filename, 'rt') as fin:
        for line in fin:
            yield line


def extract_linked(infile, outfile):
    cui_wiki = {}
    pool = mp.Pool(ncpu)
    for cui_wiki in pool.imap_unordered(_extract_linked, wikidata_iter(infile)):
        cui, wiki = cui_wiki
        if cui == None:
            continue
        else:
            cui_wiki[cui] = wiki

    print(len(cui_wiki))

    with open(outfile, 'w') as fout:
        json.dump(cui_wiki, fout)



def _extract_linked(line):
    try:
        #print(line)
        obj = json.loads(line.rstrip(',\n'), encoding='utf-8')
        claims = obj['claims']
        linked = False
        for k, v in codes.items():
            if v in claims:
                linked = True
                val = claims[v][0]['mainsnak']['datavalue']['value']
                obj[k] = val
        if linked:
            cui = obj.get(extractioncode, None)
            if cui and 'enwiki' in obj['sitelinks']:
                wiki = obj['sitelinks']['enwiki']['title']
                return cui,  wiki
    except:
        pass
    return None, None


def slow_extract(ifile, ofile):
    with gzip.open(ofile, 'w') as fout:
        with gzip.open(ifile, 'rt') as fin:
            count = 0
            for line in fin:
                count += 1
                if count % 100000 == 0:
                    print(count, code_counts)
                try:
                    #print(line)
                    obj = json.loads(line.rstrip(',\n'), encoding='utf-8')
                    claims = obj['claims']
                    linked = False
                    for k, v in codes.items():
                        if v in claims:
                            linked = True
                            val = claims[v][0]['mainsnak']['datavalue']['value']
                            obj[k] = val
                            code_counts[k] += 1
                    if linked:
                        cui = obj.get('meshdesc', None)
                        if not cui:
                            continue
                        if 'enwiki' not in obj['sitelinks']:
                            continue
                        wiki = obj['sitelinks']['enwiki']['title']
                        cui_wiki[cui] = wiki.replace(' ', '_')
                        #fout.write(json.dumps(obj).encode('utf-8'))
                        #fout.wirte('\n')
                except Exception as e:
                    print(str(e))
    print(len(cui_wiki))

    with open(ofile, 'w') as fout:
        json.dump(cui_wiki, fout)

    print(code_counts)

if __name__ == '__main__':
    extract_linked(infile, outfile)
