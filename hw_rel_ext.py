#!/usr/bin/env python
# coding: utf-8

# # Homework and bake-off: Relation extraction using distant supervision

# In[70]:


__author__ = "Bill MacCartney and Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Baselines](#Baselines)
#   1. [Hand-build feature functions](#Hand-build-feature-functions)
#   1. [Distributed representations](#Distributed-representations)
# 1. [Homework questions](#Homework-questions)
#   1. [Different model factory [1 points]](#Different-model-factory-[1-points])
#   1. [Directional unigram features [1.5 points]](#Directional-unigram-features-[1.5-points])
#   1. [The part-of-speech tags of the "middle" words [1.5 points]](#The-part-of-speech-tags-of-the-"middle"-words-[1.5-points])
#   1. [Bag of Synsets [2 points]](#Bag-of-Synsets-[2-points])
#   1. [Your original system [3 points]](#Your-original-system-[3-points])
# 1. [Bake-off [1 point]](#Bake-off-[1-point])

# ## Overview
# 
# This homework and associated bake-off are devoted to developing really effective relation extraction systems using distant supervision. 
# 
# As with the previous assignments, this notebook first establishes a baseline system. The initial homework questions ask you to create additional baselines and suggest areas for innovation, and the final homework question asks you to develop an original system for you to enter into the bake-off.

# ## Set-up
# 
# See [the first notebook in this unit](rel_ext_01_task.ipynb#Set-up) for set-up instructions.

# In[71]:


import numpy as np
import os
import rel_ext
from sklearn.linear_model import LogisticRegression
import utils


# As usual, we unite our corpus and KB into a dataset, and create some splits for experimentation:

# In[72]:


rel_ext_data_home = os.path.join('data', 'rel_ext_data')


# In[73]:


corpus = rel_ext.Corpus(os.path.join(rel_ext_data_home, 'corpus.tsv.gz'))


# In[74]:


kb = rel_ext.KB(os.path.join(rel_ext_data_home, 'kb.tsv.gz'))


# In[75]:


dataset = rel_ext.Dataset(corpus, kb)


# You are not wedded to this set-up for splits. The bake-off will be conducted on a previously unseen test-set, so all of the data in `dataset` is fair game:

# In[76]:


splits = dataset.build_splits(
    split_names=['tiny', 'train', 'dev'],
    split_fracs=[0.01, 0.79, 0.20],
    seed=1)


# In[77]:


splits


# ## Baselines

# ### Hand-build feature functions

# In[9]:


def simple_bag_of_words_featurizer(kbt, corpus, feature_counter):
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split(' '):
            feature_counter[word] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for word in ex.middle.split(' '):
            feature_counter[word] += 1
    return feature_counter


# In[10]:


featurizers = [simple_bag_of_words_featurizer]


# In[11]:


model_factory = lambda: LogisticRegression(fit_intercept=True, solver='liblinear')


# In[13]:


baseline_results = rel_ext.experiment(
    splits,
    train_split='train',
    test_split='dev',
    featurizers=featurizers,
    model_factory=model_factory,
    verbose=True)


# Studying model weights might yield insights:

# In[14]:


rel_ext.examine_model_weights(baseline_results)


# ### Distributed representations
# 
# This simple baseline sums the GloVe vector representations for all of the words in the "middle" span and feeds those representations into the standard `LogisticRegression`-based `model_factory`. The crucial parameter that enables this is `vectorize=False`. This essentially says to `rel_ext.experiment` that your featurizer or your model will do the work of turning examples into vectors; in that case, `rel_ext.experiment` just organizes these representations by relation type.

# In[15]:


GLOVE_HOME = os.path.join('data', 'glove.6B')


# In[16]:


glove_lookup = utils.glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))


# In[17]:


def glove_middle_featurizer(kbt, corpus, np_func=np.sum):
    reps = []
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split():
            rep = glove_lookup.get(word)
            if rep is not None:
                reps.append(rep)
    # A random representation of the right dimensionality if the
    # example happens not to overlap with GloVe's vocabulary:
    if len(reps) == 0:
        dim = len(next(iter(glove_lookup.values())))                
        return utils.randvec(n=dim)
    else:
        return np_func(reps, axis=0)


# In[18]:


glove_results = rel_ext.experiment(
    splits,
    train_split='train',
    test_split='dev',
    featurizers=[glove_middle_featurizer],    
    vectorize=False, # Crucial for this featurizer!
    verbose=True)


# With the same basic code design, one can also use the PyTorch models included in the course repo, or write new ones that are better aligned with the task. For those models, it's likely that the featurizer will just return a list of tokens (or perhaps a list of lists of tokens), and the model will map those into vectors using an embedding.

# ## Homework questions
# 
# Please embed your homework responses in this notebook, and do not delete any cells from the notebook. (You are free to add as many cells as you like as part of your responses.)

# ### Different model factory [1 points]
# 
# The code in `rel_ext` makes it very easy to experiment with other classifier models: one need only redefine the `model_factory` argument. This question asks you to assess a [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
# 
# __To submit:__ A wrapper function `run_svm_model_factory` that does the following: 
# 
# 1. Uses `rel_ext.experiment` with the model factory set to one based in an `SVC` with `kernel='linear'` and all other arguments left with default values. 
# 1. Trains on the 'train' part of `splits`.
# 1. Assesses on the `dev` part of `splits`.
# 1. Uses `featurizers` as defined above. 
# 1. Returns the return value of `rel_ext.experiment` for this set-up.
# 
# The function `test_run_svm_model_factory` will check that your function conforms to these general specifications.

# In[19]:


from sklearn.svm import SVC
def run_svm_model_factory():
    
    ##### YOUR CODE HERE

    return rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        featurizers=featurizers,
        model_factory=lambda : SVC(kernel='linear'))


# In[20]:


def test_run_svm_model_factory(run_svm_model_factory):
    results = run_svm_model_factory()
    assert 'featurizers' in results,         "The return value of `run_svm_model_factory` seems not to be correct"
    # Check one of the models to make sure it's an SVC:
    assert 'SVC' in results['models']['adjoins'].__class__.__name__,         "It looks like the model factor wasn't set to use an SVC."    


# In[21]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_svm_model_factory(run_svm_model_factory)


# ### Directional unigram features [1.5 points]
# 
# The current bag-of-words representation makes no distinction between "forward" and "reverse" examples. But, intuitively, there is big difference between _X and his son Y_ and _Y and his son X_. This question asks you to modify `simple_bag_of_words_featurizer` to capture these differences. 
# 
# __To submit:__
# 
# 1. A feature function `directional_bag_of_words_featurizer` that is just like `simple_bag_of_words_featurizer` except that it distinguishes "forward" and "reverse". To do this, you just need to mark each word feature for whether it is derived from a subject–object example or from an object–subject example.  The included function `test_directional_bag_of_words_featurizer` should help verify that you've done this correctly.
# 
# 2. A call to `rel_ext.experiment` with `directional_bag_of_words_featurizer` as the only featurizer. (Aside from this, use all the default values for `rel_ext.experiment` as exemplified above in this notebook.)
# 
# 3. `rel_ext.experiment` returns some of the core objects used in the experiment. How many feature names does the `vectorizer` have for the experiment run in the previous step? Include the code needed for getting this value. (Note: we're partly asking you to figure out how to get this value by using the sklearn documentation, so please don't ask how to do it!)

# In[22]:


def directional_bag_of_words_featurizer(kbt, corpus, feature_counter): 
    # Append these to the end of the keys you add/access in 
    # `feature_counter` to distinguish the two orders. You'll
    # need to use exactly these strings in order to pass 
    # `test_directional_bag_of_words_featurizer`.
    subject_object_suffix = "_SO"
    object_subject_suffix = "_OS"
    
    ##### YOUR CODE HERE
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split(' '):
            feature_counter[word+subject_object_suffix] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for word in ex.middle.split(' '):
            feature_counter[word+object_subject_suffix] += 1
            
    return feature_counter


# Call to `rel_ext.experiment`:
##### YOUR CODE HERE    
directional_bow_results = rel_ext.experiment(
    splits,
    featurizers=[directional_bag_of_words_featurizer],
)


# In[23]:


print("Number of features: ", len(directional_bow_results['vectorizer'].vocabulary_))


# In[24]:


def test_directional_bag_of_words_featurizer(corpus):
    from collections import defaultdict
    kbt = rel_ext.KBTriple(rel='worked_at', sbj='Randall_Munroe', obj='xkcd')
    feature_counter = defaultdict(int)
    # Make sure `feature_counter` is being updated, not reinitialized:
    feature_counter['is_OS'] += 5
    feature_counter = directional_bag_of_words_featurizer(kbt, corpus, feature_counter)
    expected = defaultdict(
        int, {'is_OS':6,'a_OS':1,'webcomic_OS':1,'created_OS':1,'by_OS':1})
    assert feature_counter == expected,         "Expected:\n{}\nGot:\n{}".format(expected, feature_counter)


# In[25]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_directional_bag_of_words_featurizer(corpus)


# ### The part-of-speech tags of the "middle" words [1.5 points]
# 
# Our corpus distribution contains part-of-speech (POS) tagged versions of the core text spans. Let's begin to explore whether there is information in these sequences, focusing on `middle_POS`.
# 
# __To submit:__
# 
# 1. A feature function `middle_bigram_pos_tag_featurizer` that is just like `simple_bag_of_words_featurizer` except that it creates a feature for bigram POS sequences. For example, given 
# 
#   `The/DT dog/N napped/V`
#   
#    we obtain the list of bigram POS sequences
#   
#    `b = ['<s> DT', 'DT N', 'N V', 'V </s>']`. 
#    
#    Of course, `middle_bigram_pos_tag_featurizer` should return count dictionaries defined in terms of such bigram POS lists, on the model of `simple_bag_of_words_featurizer`.  Don't forget the start and end tags, to model those environments properly! The included function `test_middle_bigram_pos_tag_featurizer` should help verify that you've done this correctly.
# 
# 2. A call to `rel_ext.experiment` with `middle_bigram_pos_tag_featurizer` as the only featurizer. (Aside from this, use all the default values for `rel_ext.experiment` as exemplified above in this notebook.)

# In[26]:


def middle_bigram_pos_tag_featurizer(kbt, corpus, feature_counter):
    
    ##### YOUR CODE HERE
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for pos_bigram in get_tag_bigrams(ex.middle_POS):
            feature_counter[pos_bigram] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for pos_bigram in get_tag_bigrams(ex.middle_POS):
            feature_counter[pos_bigram] += 1
            

    return feature_counter


def get_tag_bigrams(s):
    """Suggested helper method for `middle_bigram_pos_tag_featurizer`.
    This should be defined so that it returns a list of str, where each 
    element is a POS bigram."""
    # The values of `start_symbol` and `end_symbol` are defined
    # here so that you can use `test_middle_bigram_pos_tag_featurizer`.
    start_symbol = "<s>"
    end_symbol = "</s>"
    
    ##### YOUR CODE HERE
    tags = get_tags(s)
    return list(map(lambda x: x[0] + " " + x[1], zip([start_symbol]+tags, tags+[end_symbol])))


    
def get_tags(s): 
    """Given a sequence of word/POS elements (lemmas), this function
    returns a list containing just the POS elements, in order.    
    """
    return [parse_lem(lem)[1] for lem in s.strip().split(' ') if lem]


def parse_lem(lem):
    """Helper method for parsing word/POS elements. It just splits
    on the rightmost / and returns (word, POS) as a tuple of str."""
    return lem.strip().rsplit('/', 1)  

# Call to `rel_ext.experiment`:
##### YOUR CODE HERE

pos_tag_results = rel_ext.experiment(
    splits,
    featurizers=[middle_bigram_pos_tag_featurizer]
)


# In[27]:


def test_middle_bigram_pos_tag_featurizer(corpus):
    from collections import defaultdict
    kbt = rel_ext.KBTriple(rel='worked_at', sbj='Randall_Munroe', obj='xkcd')
    feature_counter = defaultdict(int)
    # Make sure `feature_counter` is being updated, not reinitialized:
    feature_counter['<s> VBZ'] += 5
    feature_counter = middle_bigram_pos_tag_featurizer(kbt, corpus, feature_counter)
    expected = defaultdict(
        int, {'<s> VBZ':6,'VBZ DT':1,'DT JJ':1,'JJ VBN':1,'VBN IN':1,'IN </s>':1})
    assert feature_counter == expected,         "Expected:\n{}\nGot:\n{}".format(expected, feature_counter)


# In[28]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_middle_bigram_pos_tag_featurizer(corpus)


# ### Bag of Synsets [2 points]
# 
# The following allows you to use NLTK's WordNet API to get the synsets compatible with _dog_ as used as a noun:
# 
# ```
# from nltk.corpus import wordnet as wn
# dog = wn.synsets('dog', pos='n')
# dog
# [Synset('dog.n.01'),
#  Synset('frump.n.01'),
#  Synset('dog.n.03'),
#  Synset('cad.n.01'),
#  Synset('frank.n.02'),
#  Synset('pawl.n.01'),
#  Synset('andiron.n.01')]
# ```
# 
# This question asks you to create synset-based features from the word/tag pairs in `middle_POS`.
# 
# __To submit:__
# 
# 1. A feature function `synset_featurizer` that is just like `simple_bag_of_words_featurizer` except that it returns a list of synsets derived from `middle_POS`. Stringify these objects with `str` so that they can be `dict` keys. Use `convert_tag` (included below) to convert tags to `pos` arguments usable by `wn.synsets`. The included function `test_synset_featurizer` should help verify that you've done this correctly.
# 
# 2. A call to `rel_ext.experiment` with `synset_featurizer` as the only featurizer. (Aside from this, use all the default values for `rel_ext.experiment`.)

# In[29]:


from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
def synset_featurizer(kbt, corpus, feature_counter):
    
    ##### YOUR CODE HERE
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for synset in get_synsets(ex.middle_POS):
            feature_counter[synset] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for synset in get_synsets(ex.middle_POS):
            feature_counter[synset] += 1

    return feature_counter


def get_synsets(s):
    """Suggested helper method for `synset_featurizer`. This should
    be completed so that it returns a list of stringified Synsets 
    associated with elements of `s`.
    """   
    # Use `parse_lem` from the previous question to get a list of
    # (word, POS) pairs. Remember to convert the POS strings.
    wt = [parse_lem(lem) for lem in s.strip().split(' ') if lem]
    
    ##### YOUR CODE HERE
    synsets = []
    for word, tag in wt:
        for synset in wn.synsets(word, pos=convert_tag(tag)):
            synsets.append(str(synset))
    return synsets
    
def convert_tag(t):
    """Converts tags so that they can be used by WordNet:
    
    | Tag begins with | WordNet tag |
    |-----------------|-------------|
    | `N`             | `n`         |
    | `V`             | `v`         |
    | `J`             | `a`         |
    | `R`             | `r`         |
    | Otherwise       | `None`      |
    """        
    if t[0].lower() in {'n', 'v', 'r'}:
        return t[0].lower()
    elif t[0].lower() == 'j':
        return 'a'
    else:
        return None    


# Call to `rel_ext.experiment`:
##### YOUR CODE HERE    

synsets_results = rel_ext.experiment(
    splits,
    featurizers=[synset_featurizer],
    verbose=True)


# In[30]:


def test_synset_featurizer(corpus):
    from collections import defaultdict
    kbt = rel_ext.KBTriple(rel='worked_at', sbj='Randall_Munroe', obj='xkcd')
    feature_counter = defaultdict(int)
    # Make sure `feature_counter` is being updated, not reinitialized:
    feature_counter["Synset('be.v.01')"] += 5
    feature_counter = synset_featurizer(kbt, corpus, feature_counter)
    # The full return values for this tend to be long, so we just
    # test a few examples to avoid cluttering up this notebook.
    test_cases = {
        "Synset('be.v.01')": 6,
        "Synset('embody.v.02')": 1
    }
    for ss, expected in test_cases.items():   
        result = feature_counter[ss]
        assert result == expected,             "Incorrect count for {}: Expected {}; Got {}".format(ss, expected, result)


# In[31]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_synset_featurizer(corpus)


# ### Your original system [3 points]
# 
# There are many options, and this could easily grow into a project. Here are a few ideas:
# 
# - Try out different classifier models, from `sklearn` and elsewhere.
# - Add a feature that indicates the length of the middle.
# - Augment the bag-of-words representation to include bigrams or trigrams (not just unigrams).
# - Introduce features based on the entity mentions themselves. <!-- \[SPOILER: it helps a lot, maybe 4% in F-score. And combines nicely with the directional features.\] -->
# - Experiment with features based on the context outside (rather than between) the two entity mentions — that is, the words before the first mention, or after the second.
# - Try adding features which capture syntactic information, such as the dependency-path features used by Mintz et al. 2009. The [NLTK](https://www.nltk.org/) toolkit contains a variety of [parsing algorithms](http://www.nltk.org/api/nltk.parse.html) that may help.
# - The bag-of-words representation does not permit generalization across word categories such as names of people, places, or companies. Can we do better using word embeddings such as [GloVe](https://nlp.stanford.edu/projects/glove/)?
# 
# In the cell below, please provide a brief technical description of your original system, so that the teaching team can gain an understanding of what it does. This will help us to understand your code and analyze all the submissions to identify patterns and strategies.

# # Enter your system description in this cell.
# 
# 
# This is my code description attached here as requested
# 
# I have used the following features for the model:
# 1. Directional BOW features as defined above
# 2. Bigram of POS tags of the middle text as defined above
# 3. Synsets of the middle words and pos tags as defined above
# 4. Average length of the middle for both forward and backward relation (defined below)
# 5. Directional bag of POS tags for both entity mentions
# 
# We also traied different classifiers like SVM (linear and RBF), Logistic regression, Logistic regression with L1 regularization becuase the features required for a particular reltion type must be vry sparse in the whole set of features, Ada Boost classifier (with logistic regression as base estimator), Random Forest and found that the best performing model is the Random Forest classifier giving 78.9. The logistic regression model with L2 regularization achieves a score of 67.0 and the one with L1 regularization achieves a score of 67.8. We also tried the calibarated version of RF so that the class probabilities are better evaluated and found the the model still performs the good with only 0.9 points drop in performance.
# 
# Note: We also experimented with bigram features did not find any improvement for the added complexity and features.
# 
# My peak score was: 0.789
# 

# In[85]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from  sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def middle_len_featurizer(kbt, corpus, feature_counter):
    middle_len = 0
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        middle_len += len(ex.middle)
    feature_counter["middle_len_forward"] = 0
    if(len(corpus.get_examples_for_entities(kbt.sbj, kbt.obj)) > 0):
        feature_counter["middle_len_forward"] = middle_len/len(corpus.get_examples_for_entities(kbt.sbj, kbt.obj))
    
    middle_len = 0
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        middle_len += len(ex.middle)
    feature_counter["middle_len_backward"] = 0
    if(len(corpus.get_examples_for_entities(kbt.obj, kbt.sbj)) > 0):
        feature_counter["middle_len_backward"] = middle_len/len(corpus.get_examples_for_entities(kbt.obj, kbt.sbj))

    return feature_counter

def entity_pos_featurizer(kbt, corpus, feature_counter):
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for pos in get_tags(ex.mention_1_POS):
            feature_counter[pos+"_1_SO"] += 1
        for pos in get_tags(ex.mention_2_POS):
            feature_counter[pos+"_2_SO"] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for pos in get_tags(ex.mention_1_POS):
            feature_counter[pos+"_1_OS"] += 1
        for pos in get_tags(ex.mention_2_POS):
            feature_counter[pos+"_2_OS"] += 1
    return feature_counter

def directional_bag_of_words_bigrams_featurizer(kbt, corpus, feature_counter): 
    subject_object_suffix = "_SO"
    object_subject_suffix = "_OS"

    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        words = list(map(lambda x: x+subject_object_suffix, ex.middle.split(' ')))
        START_TOKEN = "<START>"
        END_TOKEN = "<END>"
        for bigram in zip([START_TOKEN]+words, words + [END_TOKEN]):
            feature_counter[str(bigram)] += 1
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        words = list(map(lambda x: x+object_subject_suffix, ex.middle.split(' ')))
        START_TOKEN = "<START>"
        END_TOKEN = "<END>"
        for bigram in zip([START_TOKEN]+words, words + [END_TOKEN]):
            feature_counter[str(bigram)] += 1
            
    return feature_counter

def find_best_model_factory():
    logistic = lambda: LogisticRegression(fit_intercept=True, solver='liblinear', random_state=42, max_iter=200)
    logistic_l1 = lambda: LogisticRegression(fit_intercept=True, solver='liblinear', random_state=42, max_iter=200, penalty='l1')
    rf = lambda: RandomForestClassifier(n_jobs=-1, random_state=42)
    rf_calibrated = lambda: CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_jobs=-1, random_state=42), method='isotonic', cv=5)
    adaboost_decision = lambda: AdaBoostClassifier(random_state=42)
    adaboost_linear = lambda: AdaBoostClassifier(base_estimator=LogisticRegression(fit_intercept=True, solver='liblinear', random_state=42, max_iter=200), random_state=42)
    svm_linear = lambda: SVC(kernel='linear')
    svm = lambda: SVC()
    models = {}
    featurizers = [synset_featurizer, middle_bigram_pos_tag_featurizer, 
                   directional_bag_of_words_featurizer, entity_pos_featurizer, middle_len_featurizer]

    best_original_system = None
    best_score = 0
    best_model = None
    for model_factory in [logistic, logistic_l1, rf, rf_calibrated, adaboost_decision, adaboost_linear, svm_linear, svm]:
        print(model_factory())
        original_system_results = rel_ext.experiment(
            splits,
            train_split='train',
            test_split='dev',
            featurizers=featurizers,
            model_factory=model_factory,
            verbose=True)
        models[model_factory()] = original_system_results
        score = original_system_results['score']
        if(score > best_score):
            best_score = score
            best_original_system = original_system_results
            best_model = model_factory()
    print(best_score, best_model)
    return best_score, best_model, best_original_system, models


# In[86]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    best_score, best_model, best_original_system, models = find_best_model_factory()


# In[33]:


# Please do not remove this comment.


# In[114]:


# Examine feature importances for RF classifier
def examine_model_weights(train_result, k=3, verbose=True):
    vectorizer = train_result['vectorizer']

    if vectorizer is None:
        print("Model weights can be examined only if the featurizers "
              "are based in dicts (i.e., if `vectorize=True`).")
        return

    feature_names = vectorizer.get_feature_names()
    for rel, model in train_result['models'].items():
        print('Highest and lowest feature weights for relation {}:\n'.format(rel))
        try:
            coefs = model.feature_importances_.toarray()
        except AttributeError:
            coefs = model.feature_importances_
        sorted_weights = sorted([(wgt, idx) for idx, wgt in enumerate(coefs)], reverse=True)
        for wgt, idx in sorted_weights[:k]:
            print('{:10.3f} {}'.format(wgt, feature_names[idx]))
        print('{:>10s} {}'.format('.....', '.....'))
        for wgt, idx in sorted_weights[-k:]:
            print('{:10.3f} {}'.format(wgt, feature_names[idx]))
        print()
if 'IS_GRADESCOPE_ENV' not in os.environ:        
    examine_model_weights(best_original_system)


# In[115]:


def find_new_relation_instances(
        splits,
        trained_model,
        train_split='train',
        test_split='dev',
        k=10,
        vectorize=True,
        verbose=True):
    train_result = trained_model
    test_split = splits[test_split]
    neg_o, neg_y = test_split.build_dataset(
        include_positive=False,
        sampling_rate=0.1)
    neg_X, _ = test_split.featurize(
        neg_o,
        featurizers=train_result['featurizers'],
        vectorizer=train_result['vectorizer'],
        vectorize=vectorize)
    # Report highest confidence predictions:
    for rel, model in train_result['models'].items():
        print('Highest probability examples for relation {}:\n'.format(rel))
        probs = model.predict_proba(neg_X[rel])
        probs = [prob[1] for prob in probs] # probability for class True
        sorted_probs = sorted([(p, idx) for idx, p in enumerate(probs)], reverse=True)
        for p, idx in sorted_probs[:k]:
            print('{:10.3f} {}'.format(p, neg_o[rel][idx]))
        print()
if 'IS_GRADESCOPE_ENV' not in os.environ:
    find_new_relation_instances(splits, best_original_system, k=10)


# ## Bake-off [1 point]
# 
# For the bake-off, we will release a test set. The announcement will go out on the discussion forum. You will evaluate your custom model from the previous question on these new datasets using the function `rel_ext.bake_off_experiment`. Rules:
# 
# 1. Only one evaluation is permitted.
# 1. No additional system tuning is permitted once the bake-off has started.
# 
# The cells below this one constitute your bake-off entry.
# 
# People who enter will receive the additional homework point, and people whose systems achieve the top score will receive an additional 0.5 points. We will test the top-performing systems ourselves, and only systems for which we can reproduce the reported results will win the extra 0.5 points.
# 
# Late entries will be accepted, but they cannot earn the extra 0.5 points. Similarly, you cannot win the bake-off unless your homework is submitted on time.
# 
# The announcement will include the details on where to submit your entry.

# In[ ]:


# Enter your bake-off assessment code in this cell. 
# Please do not remove this comment.
if 'IS_GRADESCOPE_ENV' not in os.environ:
    pass
    # Please enter your code in the scope of the above conditional.
    ##### YOUR CODE HERE



# In[ ]:


# On an otherwise blank line in this cell, please enter
# your macro-average f-score (an F_0.5 score) as reported 
# by the code above. Please enter only a number between 
# 0 and 1 inclusive. Please do not remove this comment.
if 'IS_GRADESCOPE_ENV' not in os.environ:
    pass
    # Please enter your score in the scope of the above conditional.
    ##### YOUR CODE HERE


