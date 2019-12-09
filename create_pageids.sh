gunzip -c enwiki-latest-page.csv.gz | jq --raw-input --slurp 'split("\n") | map(split("\t")) | .[0:-1] | map( { "id": .[0], "namespace": .[1], "is_redirect": .[4], "title": .[2]  }  )' > pageids.txt
