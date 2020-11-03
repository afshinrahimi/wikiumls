# WikiUMLS: Aligning UMLS to Wikipedia via Cross-lingual Neural Ranking (under construction)


# Neurally ranked candidates 

JSON file in the form of cui:[wikiid1, ....] that can be used for manual aligning of cui to wikipedia/wikidata

Download [here](https://drive.google.com/open?id=12EGrZr1KFcFS9UwFCTnf_--XUCqW1yf9).

# Dataset 

Gold values cui:wikiid partitioned into trianing, dev, and test collected from wikidata annotated by wikidata contributors

Download [here](https://drive.google.com/file/d/1W6ACUp5X4c4M0ER12CHGAUDW4m_aTfrB/view?usp=sharing).

convert wiki page id (e.g. 17524) into a wikipedia title using https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids=17524&inprop=url for one record. For batch conversion, use a Wikipedia dump.

Training, dev, and test data containing cuialias1, cuialias2, ....  \t wikialias1, wikialias2, .... \t label (0 or 1)
that can be used to train and evaluate an alignment model.

Download [here](https://drive.google.com/file/d/1Y2gbF8xpc9YhJXMEyvdymAweCdNKdHzQ/view?usp=sharing).

# UMLS to Wiki candidates BERT-style two sentence binary classification (relevant or not)

It contain: cuialias1, cuialias2, ....  \t wikialias1, wikialias2, ....
The trained model is used to make predictions on 700k UMLS CUIs against Wiki candidates.
This is the 700k * 64 candidate set, so it is big. There is no golden alignment for this set.

Please email me for this, it is big.


# Candidates

To generate candidates, UMLS CUI aliases are used as query against aliases + Wikipedia page text (if existed, many wikidata items have no wikipedia page). Index is built using https://github.com/lemire/IndexWikipedia.

Download [here](https://drive.google.com/file/d/1mYetd62m_wEMZ4L93OQyZUiAe4GyRCIm/view?usp=sharing).


# Code

Most code is used for preprocessing of the dataset, otherwise, you can use BERT to train the two sentence classification.
You can download the dataset from here, a Wikidata dump, a Wikipedia dump, and UMLS (to extract CUI aliases) or you can download the preprocessed training files with content above with 0 or 1 relevance scores to benchmark your alignment model.

# Future work
We did not use Wikipedia text for the final neural reranking, because we wanted our model to generalise on all Wikidata where Wikipedia text for most entities is not available. The relationship between entities exists both in UMLS and Wikidata, we did not use those relations for alignment, the use of which can potentially improve the alignment.

