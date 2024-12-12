#!/usr/bin/env python

# %pip  install numpy
# %pip install datasets
# %pip install ipywidgets 
# %pip install optimum
# %pip install --upgrade huggingface_hub

import torch
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import os
import huggingface_hub
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# load model
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

def extract_hidden_states(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    batch_index = 0 
    cls_index = 0
    return [hidden_state[batch_index, cls_index,:] 
                     for hidden_state in outputs.hidden_states]
def predict(hidden_state):
    logits = model.classifier(hidden_state).cpu().detach().numpy()
    pred = logits.argmax()
    p = np.exp(logits.max()) / np.sum(np.exp(logits))
    return {'pred': pred, 'p': p}
def load_dset(dataset):
    res = []
    for input_index, example in tqdm(enumerate(dataset['validation']), total=50000):
        res += [{
            'input_index': input_index,
            'layer_index': layer_index,
            'label': example['label'],
            'hidden_state': hidden_state,
            **predict(hidden_state)
        } for layer_index, hidden_state in enumerate(extract_hidden_states(example['image']))]
    pd.DataFrame(res).head(3)
def dset_shortcut(dataset):
    res = []
    # We can't use the entire dataset without running out of memory.
    # It would require more complicated data handling to solve these issues.
    # Complicating things is against the One Shot Learning philosophy,
    # so we randomly select a subset of the data.
    random_labels = set(np.random.RandomState(42).choice(list(model.config.id2label.keys()), 200))

    for input_index, example in tqdm(enumerate(dataset['validation']), total=50000):
        if input_index > 0 and input_index % 1000 == 0:
            pd.DataFrame(res).to_json(f'data/{input_index}.jsonl', lines=True, orient='records')
            res = []

        if example['label'] not in random_labels or example['image'].mode != 'RGB':
            continue
        res += [{
            'input_index': input_index,
            'label': example['label'],
            'layer_index': layer_index,
            'hidden_state': hidden_state.detach().numpy(),
            **predict(hidden_state)
        } for layer_index, hidden_state in enumerate(extract_hidden_states(example['image']))]

    if len(res) > 0:
        pd.DataFrame(res).to_json(f'data/{input_index + 1}.jsonl', lines=True, orient='records')
def get_early_exit_layer_and_pred(preds, threshold):
    for layer_index in range(len(preds) - threshold + 1):
        if len(set(preds[layer_index: layer_index + threshold])) == 1:
            return {
                'early_exit_layer':  layer_index + threshold - 1,
                'early_exit_pred':  preds[layer_index]
            }
    return {
        'early_exit_layer':  len(preds) - 1,
        'early_exit_pred':  preds[-1]
    }
def get_early_exit(df, pred_col):
    res = []
    for threshold in trange(1, df['layer_index'].max() + 1):
        df_early_exit = df.groupby(['input_index', 'label'])[pred_col].apply(list).reset_index()

        df_early_exit['early_exit_res'] = df_early_exit[pred_col].apply(
            get_early_exit_layer_and_pred,
            threshold=threshold
        )

        df_early_exit = pd.concat([
            df_early_exit,
            pd.json_normalize(df_early_exit['early_exit_res'])
        ], axis=1)
        df_early_exit['accuracy'] = df_early_exit['label'] == df_early_exit['early_exit_pred']

        res.append(df_early_exit[['accuracy', 'early_exit_layer']].mean())

    return pd.DataFrame(res)

hidden_states = extract_hidden_states(image)
hidden_states[0].shape #First dim is batch, second is # of tokens the model has made and last is the hidd state #dim

huggingface_hub.login(token='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')
dataset = load_dataset('imagenet-1k', streaming=True, use_auth_token=True)

dataset['validation']

for i,example in tqdm(enumerate(dataset['validation'])):
    extract_hidden_states(example['image'])
    break
example['label']
print(model.config.id2label[example['label']])

df = pd.concat([pd.read_json(f'data/{filename}', lines=True)
           for filename in tqdm(os.listdir('data'))])

print(df.head())
df.shape

early_exit = get_early_exit(df, 'pred')
early_exit.head()

sns.set_context('talk')
sns.set_style('whitegrid')
sns.lineplot(x = early_exit['early_exit_layer'], y = early_exit['accuracy'], marker='o')

print(df.shape, early_exit.shape)

sns.lineplot(x=df['layer_index'],y=df['pred'] == df['label'])
sns.set_context('talk')
sns.set_style('whitegrid')
sns.lineplot(x=df['layer_index'],y=df['p'])


# mean dist between layers to last layer
tqdm.pandas()

df['last_hidden_state'] = df.groupby('input_index')['hidden_state'].transform('last')
df['dist'] = df.progress_apply(lambda row: np.linalg.norm(np.array(row['hidden_state'])-
                                                                       np.array(row['last_hidden_state'])),axis=1)

df.columns

sns.lineplot(x=df['layer_index'],y=df['dist'])
# # there is a great distance. solution: linear transformation of the layers toward the last index

df_train = df.iloc[:int(len(df) * 0.8)]
df_test = df.iloc[int(len(df) * 0.8):]
len(df_train), len(df_test)

# calculate the projected model
projectors = []
for layer_index in trange(df_train['layer_index'].max() + 1):
    df_to_use = df_train[df_train['layer_index'] == layer_index]
    X = np.stack(df_to_use['hidden_state'])
    y = np.stack(df_to_use['last_hidden_state'])
    projectors.append(LinearRegression().fit(X, y))

df_test['proj_hidden_state'] = df_test.progress_apply(
    lambda row: projectors[row['layer_index']].predict([row['hidden_state']])[0],
    axis=1
)

tqdm.pandas

#Apply model on test data
df_test['proj_pred'] = df_test['proj_hidden_state'].progress_apply(
    lambda proj_hidden_state: predict(torch.Tensor(proj_hidden_state))['pred']
)

df_test['layer_index']
sns.lineplot(x=df_test['layer_index'],
             y=df_test['pred'] == df_test['label'],
             label='before')

sns.lineplot(x=df_test['layer_index'],
             y=df_test['proj_pred'] == df_test['label'],
             label='projected')

early_exit_before = get_early_exit(df_test, 'pred')
early_exit_proj = get_early_exit(df_test, 'proj_pred')

sns.lineplot(x= early_exit_before['early_exit_layer'],
             y=early_exit_before['accuracy'],
             marker='o',
             label='before')

sns.lineplot(x= early_exit_proj['early_exit_layer'],
             y=early_exit_proj['accuracy'],
             marker='o',
             label='projected')




