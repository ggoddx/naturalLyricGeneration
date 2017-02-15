import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

data_labels = []
data_samples = []
n_features = 1000

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    #print(top_feats)
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.01, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    
            

fName = "/Users/snehabhadbhade/PycharmProjects/SI695/artists.csv"

t0 = time()

with open(fName, "rU") as f:
    file_data = f.readlines()
print("No of lines", len(file_data))

for row in file_data:
    data = row.split("\t")
    label = data[0]
    label = data[0]
    lyric = data[1]
    data_labels.append(label)
    data_samples.append(lyric)

print("done in %0.3fs." % (time() - t0))



''' Use TF Idf vectorization . '''

print("Extracting tf-idf features...")

stop_words = ['instrumental', 'vocals', 'saxophone', 'ev', 'instrumentals', 'keyboards', 'jukebox', 'wo', 'es', 'and', 'the', 'you', 'your', 'can', 'is', 'in', 'me']

vectorizer = TfidfVectorizer(max_df=0.98, min_df=1,
                                   max_features=n_features, stop_words = "english")
t0 = time()

X = vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))


''' Get top n tfidf values in row and return them with their corresponding feature names.'''
top_n = 25
row_id = 0

features = vectorizer.get_feature_names()

#ch2 = SelectKBest(chi2, k=100)
#X_chi = ch2.fit_transform(X, data_labels)
#features = np.asarray([features[i] for i in ch2.get_support(indices=True)])
print("done in %fs" % (time() - t0))



''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
dfs = []
for row_id, label in enumerate(data_labels):
    print(row_id)
    if label == 'Before 2000' or label == 'After 2000':
        feats_df = top_mean_feats(X, features, grp_ids=row_id, min_tfidf=0.01, top_n=top_n)
        #feats_df = top_feats_in_doc(X, features, row_id, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    

plot_tfidf_classfeats_h(dfs)
    

