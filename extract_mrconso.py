import json
import gzip
import zipfile
from collections import Counter


input_file = 'MRCONSO.RRF.gz'
output_file = 'mrconso.json.gz'
cui_alias_file = 'cui_alias.json.gz'
bigdict = {}
all_langs = Counter()

def extract():

    with gzip.open(input_file, 'rt') as fin:
        for line in fin:
            fields = line.strip().split('|')
            #field structure https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly
            suppressible = fields[16]
            if suppressible != 'N':
                continue
            cui = fields[0]
            lang = fields[1]
            source = fields[11]
            source_cui = fields[13]
            term = fields[14]
            ispref = True if fields[6].lower() == 'y' else False


            #check if cui already exists
            record = bigdict.get(cui, None)
            if record:
                if ispref and lang == 'ENG':
                    record['name'] = term
                alias = record.get('alias', None)
                if alias:
                    l = alias.get(lang, None)
                    if l:
                        #l is a set
                        l.add(term)
                    else:
                        #create new lang:set
                        alias[lang] = {term}
                else:
                    #new record
                    record['alias'] = {lang:{term}}

                record[source] = source_cui
            else:
                record = {'alias':{lang:{term}}, source:source_cui}
                if ispref and lang == 'ENG':
                    record['name'] = term
                bigdict[cui] = record

    #convert the sets into lists because they're not serializable
    cui_alias_dict = {}
    for cui in bigdict:
        alias = bigdict[cui]['alias']
        all_aliases = set()
        for lang in alias:
            all_langs[lang] += 1
            all_aliases |= alias[lang]
            alias[lang] = list(alias[lang])
        cui_alias_dict[cui] = {'name':bigdict[cui].get('name', None), 'aliases': list(all_aliases)}

    for lang, count in all_langs.items():
        print(lang, count)
    print('#all entities', len(bigdict))
    print('dumping cui_alias dict...')
    with gzip.open(cui_alias_file, 'wt') as fout:
        json.dump(cui_alias_dict, fout)
    print('dumping all cui info...')
    with gzip.open(output_file, 'wt') as fout:
        '''
        fout.write('[\n')
        for cui in sorted(bigdict):
            record = bigdict[cui]
            fout.write(json.dumps({cui: record}) + ',\n'
        fout.write(']')
        '''
        json.dump(bigdict, fout, sort_keys=True)

def snomed_cui():
    cui_snomed = {}
    snomed_cui = {}

    with gzip.open(output_file, 'rt') as fout:
        bigdicts = json.load(fout)

    for cui, v in bigdicts.items():
        snomed_id = v.get('SNOMEDCT_US', None)
        if snomed_id:
            cui_snomed[cui] = snomed_id
    print('dic count:', len(cui_snomed))
    snomed_cui = {v:k for k, v in cui_snomed.items()}

    with open('cui_snomed.json' , 'w') as fout:
        json.dump(cui_snomed, fout)
    with open('snomed_cui.json', 'w') as fout:
        json.dump(snomed_cui, fout)

if __name__ == '__main__':
    extract()
    snomed_cui()





