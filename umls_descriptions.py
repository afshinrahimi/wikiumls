import json
import gzip

def get_umls_names(f_mrconso):
    #mrconso.json.gz a processed version of MRCONSO.RFF as json file
    with gzip.open(f_mrconso, 'r') as fin:
        mrconso = json.load(fin)
    return mrconso




def get_umls_docs(f_mrdef, f_mrconso, f_out, add_all_langs=True):
    cui_def = {}
    with open(f_mrdef, 'r') as fin:
        for line in fin:
            fields = line.strip().split('|')
            if len(fields) != 9:
                print(len(fields))
                continue
            cui = fields[0]
            desc = fields[5]
            current_desc = cui_def.get(cui, '')
            if len(desc) > len(current_desc):
                cui_def[cui] = desc
    mrconso = get_umls_names(f_mrconso)
    with open(f_out, 'w') as fout:
        for cui, val in mrconso.items():
            desc = cui_def.get(cui, "")
            pfname = mrconso.get(cui).get("name", None)
            if not pfname:
                print("no name in mrconso", cui, desc)
                continue
            aliases = set(mrconso.get(cui)["alias"]["ENG"]) - set([pfname])
            if add_all_langs:
                for l, names in mrconso.get(cui)["alias"].items():
                    if l != 'ENG':
                        aliases.update(set(names))
            new_desc = ""
            if desc:
                new_desc = pfname + ", " + desc
            if len(aliases) > 0:
                new_desc +=  " " + pfname + " is also called " + ", ".join(aliases) + "."
            item = {"title": pfname, "text": new_desc, "document_id": cui}
            fout.write(json.dumps(item))
            fout.write('\n')

if __name__ == '__main__':
    get_umls_docs('../extra/umls-install/2019AA/META/MRDEF.RRF', 'mrconso.json.gz', 'umlsdocs.json')



