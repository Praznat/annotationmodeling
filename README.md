This repo contains consensus-based unsupervised and semi-supervised methods for estimating the quality of complex annotations based on user-specified _distance functions_. These are described more formally here: https://www.ischool.utexas.edu/~ml/papers/braylan_web2020.pdf

Unlike related methods (e.g. Dawid-Skene, ZenCrowd, etc.), the methods here are applicable to _complex_ annotations: those that are not expressable as binary or categorical (multiple choice) responses. Some examples may include:
* Sequences - example response: [(4,18), (21,24), (102,112)]
* Free text - example response: "I made a repo today"
* Tree structure - example response: (S (NP I) (VP made (NP a repo) (NP today)))
* Image segments - example response: {"head":[(33,10), (89, 160)], "hand":[(20,210), (40,218)]}

The raw annotation dataset should contain columns for workerID, itemID, and annotation (named however you like). Unfortunately there are no command-line methods for running this right now, the user would need to write a python script or jupyter notebook. This is because of the dependence on the user-specified distance function, which must be a python function of the following form:

```
def distance_fn(annotation1, annotation2):
  # CODE
  return scalar_result
```

Here is an example of how you might run this on a translations dataset:

```
import pandas as pd
from nltk.translate.gleu_score import sentence_gleu
import experiments

annotation_df = pd.read_csv("translations.csv")

distance_fn = lambda x,y: 1 - (sentence_gleu([x.split(" ")], y.split(" ")) + sentence_gleu([y.split(" ")], x.split(" "))) / 2

translation_experiment = experiments.RealExperiment(eval_fn=None,
                                                    label_colname="translation",
                                                    item_colname="sentence", uid_colname="worker",
                                                    distance_fn=dist_fn)

translation_experiment.setup(annotation_df)
translation_experiment.train()
```

The result would be the dictionary objects `translation_experiment.bau_preds`, `translation_experiment.sad_preds`, `translation_experiment.mas_preds` containing the estimated best annotations per item according to the methods:
* BAU: chooses the annotation made by the worker who on average agrees most with consensus over the whole dataset.
* SAD: chooses the annotation closest to all other annotations for each item (like majority vote).
* MAS: chooses the annotation estimated to be best by a probabilistic model that considers both within-item consensus and worker average consensus.

