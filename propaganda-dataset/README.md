##### README file for the PTC corpus

VERSION: 2

Contents

1. About the Propaganda Techniques Corpus (PTC)
2. Tasks
3. Data format
4. Changes in Version 2


About the Propaganda Techniques Corpus (PTC)
--------------------------------------------

PTC is a corpus of propagandistic techniques at fine-grained level. 

The corpus includes 500 articles (350k tokens) from 48 news outlets. It was 
manually-annotated by six professional annotators (both unitized and labeled) 
considering 18 propaganda techniques:

* Loaded Language
* Name Calling,Labeling
* Repetition
* Exaggeration,Minimization
* Doubt
* Appeal to fear-prejudice
* Flag-Waving
* Causal Oversimplification
* Slogans
* Appeal to Authority
* Black-and-White Fallacy
* Thought-terminating Cliches
* Whataboutism
* Reductio ad Hitlerum
* Red Herring
* Bandwagon
* Obfuscation,Intentional Vagueness,Confusion
* Straw Men


Tasks
--------------------------------------------
PTC enables for the development of automatic models for propaganda techniques 
identification, i.e. multi-class setting (FLC). Furthermore, a binary task 
consisting in determining whether a sentence contains any propaganda technique
is provided (SLC). 

Data format
--------------------------------------------

The corpus includes the following folders:

train-articles: articles composing the training set in plain text format. 
train-labels-FLC: training set gold labels for task FLC
train-labels-SLC: training set gold labels for task SLC (one file per article)
dev-articles: articles composing the development set in plain text. 
dev-template-output-SLC: template for the output files for task SLC. 

and the following files:
train.task-SLC.labels: training set gold labels for task SLC in one file
dev.template-output-SLC.out: development set template output in one file

FILE FORMAT

- train-articles/dev-articles: raw content of the articles in plain text 
format. The articles are retrieved with newspaper3k library. The title is 
on the first row, followed by an empty row. The content of the article 
starts from the third row, one sentence per line (line splitting is 
performed automatically with nltk sentence splitter) 

- train-labels-FLC: tab-separated file with one propaganda technique per 
line with the following information: 

id   technique    begin_offset     end_offset

where id is the identifier of the article, technique is one out of the 18
techniques, begin_offset is the character where the covered span begins and 
end_offset is the character where the covered span ends.

The naming of the pair of files is:
- article[unique_id].txt for the plain-text file 
- article[unique_id].labels.tsv for the annotations files 

We include three subfolders: train (350 articles), dev (60 articles).

We provide the following files for the SLC task:
 -  article[unique_id].task-SLC.labels

With the following format:

article_id	sentence_id	label

where article_id and sentence_id are the identifiers of the article and the sentence 
(the first sentence has id 1) and label={propaganda/non-propaganda}
The template files in folder dev-template-output-SLC have the same format where 
label is replaced with ?. These files might be used to perform a submission by 
replacing ? with {propaganda/non-propaganda}.


Changes in Version 2
--------------------------------------------

Due to mismatch in the encoding/decoding of some non-standard characters between 
the software used for the annotations and standard Python functions to read the 
content of the articles, a very small number of annotations in 7 documents were 
shifted by 1-2 chars:

train-labels-FLC/article111111122.task-FLC.labels
train-labels-FLC/article111111131.task-FLC.labels
train-labels-FLC/article755170235.task-FLC.labels
train-labels-FLC/article756114837.task-FLC.labels
train-labels-FLC/article769752554.task-FLC.labels
train-labels-FLC/article769962236.task-FLC.labels
train-labels-FLC/article696246189.task-FLC.labels

As a consequence, one annotation in the following file had to be corrected as well:

train-labels-SLC/article696246189.task-SLC.labels

This new version (Version 2) of the corpus integrates the changes in 
these 8 files (the rest are unchanged).
