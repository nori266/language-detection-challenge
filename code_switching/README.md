## Data

For code switching we will focus on Spanish and English, and the data provided is derived from http://www.care4lang.seas.gwu.edu/cs2/call.html. However, instead of downloading the data
from this link, refer to all data inside `code_switching/data`, since we are using a modified version.

This data is a collection of tweets, in particular you have three files for the training set and three for the validation set:

- offsets_mod.tsv
- tweets.tsv
- data.tsv

The first file has the id information about the tweets, together with the tokens positions and the gold labels. The second has the ids and the actual tweet text, and the third
has the combination of the previous files, with the tokens of each sentence and the gold labels associated. More specifically, the columns are:

* `offsets_mod.tsv`:

```
tweet_id, user_id, start, end, gold label
```

* `tweets.tsv`

```
tweet_id, user_id, tweet text
```

* `data.tsv`:

```
tweet_id, user_id, start, end, token, gold label
```

The gold labels can be one of three:

* en
* es
* other

For this exercise, we require you to implement a BILSTM tagger to solve this task. You can use whatever deep learning framework you wish (Tensorflow, Pytorch, Theano, Dynet, etc.) You can take a look at the tutorial below on sequence models as a starting point.

## Suggested reading

https://homes.cs.washington.edu/~nasmith/papers/jaech+mulcaire+hathi+ostendorf+smith.lics16.pdf
http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html 
