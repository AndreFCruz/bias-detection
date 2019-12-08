from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
import numpy as np

preds_path = '/Users/andre/Documents/hyperpartisan-news-detection/preds/prediction.txt'
truth_path = '/Users/andre/Documents/hyperpartisan-news-detection/generated_data/SpaCy.54.norml2.npz'

y_test = np.load(truth_path)['Y'].flatten()
y_score = np.empty(y_test.shape)

with open(preds_path, 'r') as f:
    lines = f.readlines()
    data = [l.split(' ') for l in lines]
    for i, (_, art_class, art_score) in enumerate(data):
        art_score = float(art_score)
        if art_class == 'true':
            y_score[i] = (art_score / 2) + 0.5
        elif art_class == 'false':
            y_score[i] = 0.5 - (art_score / 2)
        else:
            raise Exception('Invalid article class "{}"'.format(art_class))

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

## NOTE let's just assume this works but I didn't properly test it
precision, recall, _ = precision_recall_curve(y_test, y_score)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()