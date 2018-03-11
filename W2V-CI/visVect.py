from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager


class visVect:
    def __init__(self, vectors, words):
        font = matplotlib.font_manager.FontProperties(fname='./ipag.ttc')
        FONT_SIZE = 20
        self.TEXT_KW = dict(fontsize=FONT_SIZE, fontweight='bold', fontproperties=font)
        self.data = vectors
        self.words = words
        
    def plot(self, query, nbest = 15):
        if ', ' not in query:
            words = 
        else:
            words = query.split(', ')
            print ', '.join(words)
        mat = w2v.get_vectors(self.data)
        word_indexes = [w2v.get_word_index(self.data, w) for w in words]
        if word_indexes == [-1]:
            print 'not in vocabulary'
            return

        # do PCA
        X = mat[word_indexes]
        pca = PCA(n_components=2)
        pca.fit(X)
        print pca.explained_variance_ratio_
        X = pca.transform(X)
        xs = X[:, 0]
        ys = X[:, 1]

        # draw
        plt.figure(figsize=(12,8))
        plt.scatter(xs, ys, marker = 'o')
        for i, w in enumerate(words):
            plt.annotate(
                w.decode('utf-8', 'ignore'),
                xy = (xs[i], ys[i]), xytext = (3, 3),
                textcoords = 'offset points', ha = 'left', va = 'top',
                **self.TEXT_KW)

        plt.show()




# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("veh b4 dish")
plt.show()


X = model[model.wv.vocab]
tsne = TSNE(n_components=2)
result = tsne.fit_transform(X)

# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("veh b4 dish")
plt.show()








vocab = list(model.wv.vocab)
X = model[vocab]
print(X[0])
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()



def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
