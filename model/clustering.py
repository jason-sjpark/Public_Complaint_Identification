from gensim.models import Word2Vec
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering


def read_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


train_df = read_data('2019_6.txt')

from konlpy.tag import Okt

okt = Okt()


def tokenizing(docs):
    return ['/'.join(t) for t in okt.pos(docs, norm=True, stem=True)]


train_pos = []
for row in train_df:
    try:
        train_pos0 = tokenizing(row[2])
        train_pos.append(train_pos0)

    except:
        pass

sentence = []
for tokens in train_pos:
    import nltk

    text = nltk.Text(tokens, name='NMSC')
    len(set(text.tokens))
    text.vocab().most_common(200)
    most_words = open('많이 나온 단어.txt', 'w', encoding='utf-8')
    for word in text.vocab().most_common(10):
        most_words.write(str(word) + '\n')
    most_words.close()

    selected_words = [f[0] for f in text.vocab().most_common(3000)]
    stop_words = ['하다/Verb', '?/Punctuation', '에/Josa', '이/Josa', '있다/Adjective', '가/Josa', './Punctuation', '되다/Verb',
                  '도/Josa', '들/Suffix',
                  '는/Josa', '은/Josa', '../Punctuation', '을/Josa', '??/Punctuation', '님/Suffix', '으로/Josa', '에서/Josa',
                  '거/Noun', '다/Adverb',
                  '거/Noun', '로/Josa', '를/Josa', '의/Josa', '때/Noun', '나/Noun', '만/Josa', 'ㅠㅠ/KoreanParticle', '1/Number',
                  '이/Determiner', '인데/Josa',
                  '2/Number', '좀/Noun', '뭐/Noun', '것/Noun', '진짜/Noun', '부산/Noun', '왜/Noun', '고/Josa', '!/Punctuation',
                  '오늘/Noun', '과/Josa',
                  ')/Punctuation', '랑/Josa', '이/Noun', '면/Josa', '한/Josa', '나/Josa', '좀/Noun', '내/Noun', '개/Noun',
                  '해주다/Verb', '까지/Josa', '대/Suffix', '3/Number',
                  '..?/Punctuation', 'ㅠㅠㅠ/KoreanParticle', '하/Suffix', '요/Josa', '더/Noun', '하나/Noun', '이랑/Josa',
                  '근데/Adverb', '아/Josa', '야/Josa', '인/Josa', '중/Noun', '4/Number',
                  '게/Josa', '한테/Josa', '애/Noun', '???/Punctuation', 'ㅋㅋ/KoreanParticle', '..../Punctuation',
                  '.../Punctuation', '"/Punctuation']
    this = []
    for w in selected_words:
        if w not in stop_words:
            real = w.split('/')
            this.append(real[0])
    sentence.append(this)


print('sentence: ', sentence)

m = Word2Vec(sentence, size=10, min_count=1, sg=1)
def vectorizer(sent, m):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else:
                vec = np.add(vec, m[w])
            numw += 1
        except:
            pass

    return np.asarray(vec) / numw


l = []
for i in sentence:
    l.append(vectorizer(i, m))
l2 = []
for i in l:
    if i.shape != (0,):
        l2.append(i)
    else:
        continue






X = np.zeros((len(l2),10))
for i in range(len(l2)):
    X[i] = l2[i]


import matplotlib.pyplot as plt
'''
wcss = []
for i in range(1, 50):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 50), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''

n_clusters = 6  ###
clf = KMeans(n_clusters=n_clusters,
             max_iter=100,
             init='k-means++',
             n_init=1)
labels = clf.fit_predict(X)
print(labels)
'''
for index, sentence in enumerate(sentence):
    print(str(labels[index]) + " : " + str(sentence))
'''

pca = PCA(n_components=6).fit(X)  ###
coords = pca.transform(X)
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]
plt.scatter(coords[:, 0], coords[:, 1], c=colors)
centroids = clf.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c="#444d61")
plt.show()

Z = hierarchy.linkage(X, 'ward')
dn = hierarchy.dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.xlim(0, 100)
plt.ylim(0, 0.05)
plt.show()

hc = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  ###
y_hc = hc.fit_predict(X)
print(y_hc)
'''
for index, sentence in enumerate(sentence):
    print(str(y_hc[index]) + ":" + str(sentence))
'''