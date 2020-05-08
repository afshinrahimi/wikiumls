# WikiUMLS: Aligning UMLS to Wikipedia via Cross-lingual Neural Ranking (under construction)


# Neurally ranked candidates 

JSON file in the form of cui:[wikiid1, ....] that can be used for manual aligning of cui to wikipedia/wikidata

Download [here](https://uq-my.sharepoint.com/:u:/g/personal/uqarahi2_uq_edu_au/Ed8ZPUArBKFNqZyhFsE7l0sB2Xop-ZLE1c9HfAljSdwMsg?e=nP2UAC).

# Dataset 

Gold values cui:wikiid partitioned into trianing, dev, and test collected from wikidata annotated by wikidata contributors

Download [here](https://uq-my.sharepoint.com/:u:/g/personal/uqarahi2_uq_edu_au/EYbzkYFxraJKrUaEZe-GG4IBHLl05a6d1KQqN-SMXDOJJw?e=TuNrm1).

Training, dev, and test data containing cuialias1, cuialias2, ....  \t wikialias1, wikialias2, .... \t label (0 or 1)
that can be used to train and evaluate an alignment model.

Download [here](https://uq-my.sharepoint.com/:u:/g/personal/uqarahi2_uq_edu_au/EZcr7S60QaVAn1q1YCfjRYwBiqcrcidrb8wSfX7PnWcoPQ?e=3ee5sN).

# UMLS to Wiki candidates BERT-style two sentence binary classification (relevant or not)

It contain: cuialias1, cuialias2, ....  \t wikialias1, wikialias2, ....
The trained model is used to make predictions on 700k UMLS CUIs against Wiki candidates.
This is the 700k * 64 candidate set, so it is big. There is no golden alignment for this set.

Download [here](https://uq-my.sharepoint.com/:u:/g/personal/uqarahi2_uq_edu_au/EdfgcJf31u1IjJ6PGw9ADt0BdIfB_UOtY-4wh50Qy3An-g?e=ACBshg).


# Candidates

To generate candidates, UMLS CUI aliases are used as query against aliases + Wikipedia page text (if existed, many wikidata items have no wikipedia page). Index is built using https://github.com/lemire/IndexWikipedia.

Download [here](https://uq-my.sharepoint.com/:u:/g/personal/uqarahi2_uq_edu_au/EfJiV-Y6f0FAjLVNp7DU8dgBbsUVcDDaWQoQk6RBFNAJ6A?e=LHDT3P).


# Code

Most code is used for preprocessing of the dataset, otherwise, you can use BERT to train the two sentence classification.
You can download the dataset from here, a Wikidata dump, a Wikipedia dump, and UMLS (to extract CUI aliases) or you can download the preprocessed training files with content above with 0 or 1 relevance scores to benchmark your alignment model.

# Future work
We did not use Wikipedia text for the final neural reranking, because we wanted our model to generalise on all Wikidata where Wikipedia text for most entities is not available. The relationship between entities exists both in UMLS and Wikidata, we did not use those relations for alignment, the use of which can potentially improve the alignment.

