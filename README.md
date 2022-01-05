# TransMorph
Implementation of "TransMorph: A Transformer Based Morphological Disambiguator for Turkish"

# UnambiguousWordAnalysis Dataset
In order to create an unambiguous data set the unambiguous sentences at Hürriyet News data set and BOUN Web corpus are combined .
These two data sets are used to filter sentences that are composed of only unambiguous words. 
Each line starts with a token or tag, then one lemma+tag analysis, separated by whitespace. 
We created the analyses using Kemal Oflazer's finite state transducers. 11842 unambiguous sentences frm Hurriyet News Data and 298703 unambiguous sentences 
from BOUN Web Corpus are combined. 310109 sentences are obtained after dropping duplicate sentences. 

Moreover, the sentences and their morphological analysis of each sentence are presented in UnambigousSentences file. Morphological analysis of words are seperated with Eow token and lemma is seperated with Eor token from analysis.
