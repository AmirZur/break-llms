import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy
import scipy.spatial.distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import warnings


pd.set_option('max_colwidth', 999)
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)


color_types = ["#58b5e1", "#782857", "#4ae182", "#ff66b2", "#048a37", "#ad0599", "#51f310", "#9620fc", "#a3c541", "#3239d6"]


def load_annotated_dataset(filename, tokenizer, model=None, layers=(1, 6, 12)):
    df = pd.read_csv(filename, index_col=0)

    # Remove cases without meaning annotations:
    df = df[(df.meaning.isnull() == False)]

    # Check that all sentences contain "break":
    df.sentence.str.contains(r"break\b").value_counts()

    # Make sure no examples need to be trimmed:
    df.sentence.apply(lambda x: check_no_trimming(x, tokenizer))

    break_ids = hf_encode(" break", tokenizer)[0]

    # Tokenizer indices:
    df['ids'] = df['sentence'].apply(lambda x: get_indices(x, tokenizer))

    if model is not None:

        df[f'reps'] = df['ids'].apply(lambda x: hf_represent(x, model))

        for layer in layers:

            df[f'break_rep_layer{layer}'] = df.apply(
                lambda row: get_break_reps(row, break_ids, layer=layer), axis=1)

        df.drop('reps', inplace=True, axis=1)

    return df


def load_reps(weights_name):
    output_dirname = "reps"
    if not os.path.exists(output_dirname):
        os.mkdir(output_dirname)

    output_filename = os.path.join(
        output_dirname,
        f"{weights_name.replace('/', '_')}_df.pickle")
    with open(output_filename, "rb") as f:
        df = pickle.load(f)
    return df


def check_no_trimming(s, tokenizer):
    toks = tokenizer.tokenize(s)
    if len(toks) >= 512:
        warnings.warn(f"Need to trim: {s}")
        break_index = s.index("break")
        s = s[: break_index + len("break")]
        toks = tokenizer.tokenize(s)
        if len(toks) >= 512:
            warnings.warn(f"\tWarning: Example still too long! {len(toks)} tokens")


def find_sublist_indices(sublist, mainlist):
    indices = []
    length = len(sublist)
    for i in range(0, len(mainlist)-length+1):
        if mainlist[i: i+length].equal(sublist):
            indices.append((i, i+length))
    return indices


def get_indices(text, tokenizer):
    ids = hf_encode(text, tokenizer, add_special_tokens=True)
    return ids[0, -512: ]


def get_reps(ids, model):
    return hf_represent(ids.unsqueeze(0), model).squeeze(0)


def hf_encode(text, tokenizer, add_special_tokens=False):
    encoding = tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors='pt')
    if encoding.shape[1] == 0:
        text = tokenizer.unk_token
        encoding = torch.tensor([[tokenizer.vocab[text]]])
    return encoding


def hf_represent(ids, model):
    with torch.no_grad():
        reps = model(ids.unsqueeze(0), output_hidden_states=True)
        # Tuple of reps of shape (batch_size, seq_length, model_dim)
        return reps.hidden_states


def get_break_reps(row, break_ids, layer=1):
    ids = row['ids']
    reps = row['reps'][layer]
    offsets = find_sublist_indices(break_ids, ids.squeeze(0))
    for (start, end) in offsets:
        pooled = mean_pooling(reps[:, start: end])
        return pooled


def mean_pooling(hidden_states):
    #_check_pooling_dimensionality(hidden_states)
    return torch.mean(hidden_states, axis=1)


def get_color(s, meaning_types):
    i = meaning_types.index(s)
    if i < len(color_types):
        return color_types[meaning_types.index(s)]
    else:
        return "gray"


def tsne_viz(df, weights_name, colname, labelcol='meaning', full_sentence=False):
    X = torch.vstack(list(df[colname].values))
    rep_df = pd.DataFrame(X.numpy())

    if full_sentence:
        plt.rc('text', usetex=False)
    else:
        plt.rc('text', usetex=True)

    def ann_trans(row):
        label = row[labelcol].replace("_", "-")
        if not full_sentence:
            trans = row['construction']
            if trans == 'unaccusative':
                label = r'\framebox{{{}}}'.format(label)
            elif trans == 'unergative':
                label = r'\underline{{{}}}'.format(label)
            label = r'\textbf{{{}}}'.format(label)
        return label

    rep_df.index = df.apply(ann_trans, axis=1)

    if full_sentence:
        ext = "pdf"
        figsize = (75, 75)
    else:
        figsize = (8.5 * 3, 11 * 3)
        ext = "pdf"

    if not os.path.exists("fig"):
        os.mkrdir("fig")

    output_filename = os.path.join("fig", f"{weights_name}-{colname}.{ext}")

    _tsne_viz_util(
        rep_df,
        colors=list(df.meaning_color.values),
        figsize=figsize,
        output_filename=output_filename,
        random_state=42)


def _tsne_viz_util(df, colors=None, output_filename=None, figsize=(40, 50), random_state=None):
    # Colors:
    vocab = df.index
    if not colors:
        colors = ['black' for i in vocab]
    # Recommended reduction via PCA or similar:
    n_components = 50 if df.shape[1] >= 50 else df.shape[1]
    dimreduce = PCA(n_components=n_components, random_state=random_state)
    X = dimreduce.fit_transform(df)
    # t-SNE:
    tsne = TSNE(n_components=2, random_state=random_state)
    tsnemat = tsne.fit_transform(X)
    # Plot values:
    xvals = tsnemat[: , 0]
    yvals = tsnemat[: , 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(xvals, yvals, marker='', linestyle='')

    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        try:
            ax.annotate(word, (x, y), fontsize=8, color=color)
        except UnicodeDecodeError:  ## Python 2 won't cooperate!
            pass
    plt.axis('off')
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()


def glove2dict(src_filename):
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float64)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


def cosine(u, v):
    return scipy.spatial.distance.cosine(u, v)


def neighbors(word, df, distfunc=cosine):
    if word not in df.index:
        raise ValueError('{} is not in this VSM'.format(word))
    w = df.loc[word]
    dists = df.apply(lambda x: distfunc(w, x), axis=1)
    return dists.sort_values()
