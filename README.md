# WikiUMLS: Aligning UMLS to Wikipedia via Cross-lingual Neural Ranking (under construction)




# Dataset 

Gold values Mesh Descriptor ID:WikiTitle collected from wikidata annotated by wikidata contributors are available [here](https://github.com/afshinrahimi/wikiumls/blob/master/data/meshdesc_wiki.json). To convert MeSH descriptor ids into UMLS CUIs use MRCONSO.RRF file which is available [here](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html). To map between Wikipedia pages and ids use a Wikipedia dump available [here](https://dumps.wikimedia.org/enwiki/latest/)

For each Wiki page aliases are extracted from Wikidata, and for each UMLS CUI aliases are extracted from MRCONSO.RFF file. Then, for each CUI query, 64 candidate pages are retrieved from Wikipedia using Lucene. A two-sentence binary classification dataset is built for CUIs and their candidates and a label is assigned (1 if the candidate is correct; 0 otherwise).



```
label \t CUI \t wikiID \t UMLS_aliases \t Wiki_aliases
1	C1565979	22524021	Screening, Whole Body , Exploration du corps entier , Screenings, Whole Body , Ganzkörperscreening , Helkroppsscreening , Whole Body Screenings , Whole Body Screening , celotělový screening	whole body imaging , imagerie du corps entier
0	C1565979	13073624	Screening, Whole Body , Exploration du corps entier , Screenings, Whole Body , Ganzkörperscreening , Helkroppsscreening , Whole Body Screenings , Whole Body Screening , celotělový screening	Whole-body nuclear scanning
```


Training, dev, and test data containing are available [here](https://drive.google.com/file/d/1Y2gbF8xpc9YhJXMEyvdymAweCdNKdHzQ/view?usp=sharing). The labels for the test set are removed from the test file and are available in meshdesc_wiki.json file.
<!-- Download [here](https://drive.google.com/file/d/1W6ACUp5X4c4M0ER12CHGAUDW4m_aTfrB/view?usp=sharing). -->



# 64 Generated Candidates

To generate candidates, UMLS CUI aliases are used as query against aliases + Wikipedia page text (if existed, many wikidata items have no wikipedia page). Index is built using https://github.com/lemire/IndexWikipedia.

Download [here](https://drive.google.com/file/d/1mYetd62m_wEMZ4L93OQyZUiAe4GyRCIm/view?usp=sharing).


# Neurally reranked Wikipedia pages for 700k UMLS CUIs

JSON file in the form of cui:[wikiid1, wikiid2, ....] that can be used for manual alignment of cui to wikipedia/wikidata. These are the result of running our model on all UMLS cocepts covered in SNOMED. We have not evaluated our model on this set as no ground truth exists.

Download [here](https://drive.google.com/open?id=12EGrZr1KFcFS9UwFCTnf_--XUCqW1yf9).

# Code

Most of the code is used for preprocessing of the dataset (Wikipedia, Wikidata, and MRCONSO.RFF), otherwise, you can use BERT to train a two sentence classification model on the dataset available here.


# Future work
We did not use Wikipedia text for the final neural reranking, because we wanted our model to generalise on all Wikidata where Wikipedia text for most entities is not available. The relationship between entities exists both in UMLS and Wikidata, we did not use those relations for alignment, the use of which can potentially improve the alignment.

