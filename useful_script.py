# -*- coding: utf-8 -*-
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS

class HarryPotter(object):

    def __init__(self):
        self.harry_potter_titles = ['philosophersStone.txt', 'chamberOfSecrets.txt', 'prisonerOfAzkaban.txt', 'gobletOfFire.txt', 'orderOfThePhoenix.txt', 'halfBloodPrince.txt', 'deathlyHallows.txt']

        self.chapters_regex = re.compile('^CHAPTER\s[0-9A-Z]+\s-\s.*?\n')

        self.chapter_list = []
        self.chapter_texts = []
        self.book_list = []
        self.clusters = None

        self.tfidf_vectorizer = TfidfVectorizer(max_df = 0.75, max_features = 400000, min_df = 0.05, stop_words='english', use_idf=True, tokenizer=self.tokenize, ngram_range=(1,4), decode_error='ignore')
        self.NUM_CLUSTERS = 25
        self.km = KMeans(n_clusters = self.NUM_CLUSTERS)
        self.tfidf_matrix = None
        self.terms = None
        self.chapter_frame = None

        self.cluster_colours = {'philosophersStone' : '#7fc97f', 'chamberOfSecrets' : '#beaed4', 'prisonerOfAzkaban' : '#fdc086', 'gobletOfFire' : '#ffff99', 'orderOfThePhoenix' : '#386cb0', 'halfBloodPrince' : '#f0027f', 'deathlyHallows' : '#bf5b17'}

    def split_into_chapters(self, book):
        transform = lambda x: ' '.join(x.split('-')[1:]).strip().lower().decode('utf-8', 'ignore').encode('ascii', 'ignore')
        current_text = ""
        current_chapter_name = ""
        chapter_dictionary = {}
        for line in book:
            if re.match(self.chapters_regex, line):
                chapter_dictionary[current_chapter_name] = current_text.decode('utf-8', 'ignore').encode('ascii', 'ignore')
                current_chapter_name = transform(line)
                current_text = ""
            else:
                current_text += ' ' + line
        chapter_dictionary[current_chapter_name] = current_text.decode('utf-8', 'ignore').encode('ascii', 'ignore')
        ## Remove first blank bit
        del chapter_dictionary['']
        return chapter_dictionary

    def generate_chapter_and_text_lists(self):
        books_as_lists = [open(entry, 'rb').readlines() for entry in self.harry_potter_titles]
        chapter_dictionaries = dict((book_title.rstrip('txt').rstrip('.'), self.split_into_chapters(book)) for book, book_title in zip(books_as_lists, self.harry_potter_titles))
        chapter_lookup_dictionary = dict((chapter_title, book_title) for book_title, book_dict in chapter_dictionaries.items() for chapter_title in book_dict.keys())
        for book_title, book_dict in chapter_dictionaries.items():
            for chapter_title, chapter_text in book_dict.items():
                self.chapter_list.append(chapter_title)
                self.chapter_texts.append(chapter_text)
                self.book_list.append(book_title)

    def tokenize(self, text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text.decode('utf8', 'ignore')) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token) and "'" not in token:
                filtered_tokens.append(token)
        return filtered_tokens

    def perform_tfidf(self):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chapter_texts)
        self.terms = self.tfidf_vectorizer.get_feature_names()

    def perform_clustering(self, number_of_clusters):
        self.km.fit(self.tfidf_matrix)
        self.clusters = self.km.labels_.tolist()
        chapters = {'chapter' : self.chapter_list, 'text' : self.chapter_texts, 'cluster' : self.clusters, 'book' : self.book_list}
        self.chapter_frame = pd.DataFrame(chapters, index = [self.clusters], columns = ['chapter', 'cluster', 'text', 'book'])

    def print_clusters(self, num_terms=10):
        order_centroids = self.km.cluster_centers_.argsort()[:,::-1]
        for i in range(self.NUM_CLUSTERS):
            print "Cluster %d words: " % i
            top_words = [str(self.terms[term_index]) for term_index in order_centroids[i, :num_terms]]
            print "Top words: %s" % ','.join(top_words)
            print ""
            for book, title in zip(self.chapter_frame.ix[i]['book'], self.chapter_frame.ix[i]['chapter']):
                print "Chapter: %s, Book: %s" % (title, book)
            print ""
            
    def generate_cluster_plot_frame(self):
        MDS()
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        dist = 1 - cosine_similarity(self.tfidf_matrix)
        pos = mds.fit_transform(dist)
        xs, ys = pos[:,0], pos[:,1]

        self.cluster_plot_frame = pd.DataFrame(dict(x=xs, y=ys, label=self.clusters, chapter=self.chapter_list, book=self.book_list))


    def plot_all_clusters(self):
        groups = self.cluster_plot_frame.groupby('book')
        fig, ax = plt.subplots(figsize=(17,9))
        ax.margins(0.05)

        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, color=self.cluster_colours[name], mec='none', label=name)
            ax.set_aspect('auto')
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

        ax.legend(numpoints=1)
        for i in range(len(self.cluster_plot_frame)):
            ax.text(self.cluster_plot_frame.ix[i]['x'], self.cluster_plot_frame.ix[i]['y'], self.cluster_plot_frame.ix[i]['chapter'], size=8)

        plt.savefig('all_clusters.png')
        plt.close()

    def factors(self, n):
        return list(set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

    def median(self, factor_list):
        lst = sorted(factor_list)
        return lst[((len(lst) + 1)/2) - 1]

    def plot_all_subplots(self):
        groups = self.cluster_plot_frame.groupby('label')
        factors = self.factors(self.NUM_CLUSTERS)
        if len(factors) % 2 == 0:
            flipflop = False
            while len(factors) > 2:
                if flipflop:
                    factors.pop()
                else:
                    factors.pop(0)
                flipflop = not flipflop
            height, width = factors
        else:
            squared_factor = self.median(factors)
            height, width = squared_factor, squared_factor

        fig = plt.figure(figsize=(17,9))
        fig.suptitle("Individual clusters")
        for name, group in groups:
            ax = plt.subplot(height, width, name + 1)
            ax.set_title("Cluster %d" % (name+ 1))
            mini_group = group.groupby('book')
            for namey, mini_df in mini_group:
                ax.plot(mini_df.x, mini_df.y, marker='o', linestyle='', ms=6, color=self.cluster_colours[namey], mec='none', label=namey)
            ax.set_aspect('auto')
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
            for index in mini_df.index:
                ax.text(mini_df.ix[index]['x'], mini_df.ix[index]['y'], mini_df.ix[index]['chapter'], size=9)
        plt.savefig('all_clusters_subplot.png')
        plt.close()

    def plot_sequential_graphs(self):
        groups = self.cluster_plot_frame.groupby('label')
        for name, group in groups:
            fig, ax = plt.subplots(figsize=(17,9))
            ax.margins(0.05)
            ax.set_title("Cluster %d" % (name + 1))
            mini_group = group.groupby('book')
            for namey, mini_df in mini_group:
                ax.plot(mini_df.x, mini_df.y, marker='o', linestyle='', ms=12, color=self.cluster_colours[namey], mec='none', label=namey)
            ax.set_aspect('auto')
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
            for index in group.index:
                ax.text(group.ix[index]['x'], group.ix[index]['y'], group.ix[index]['chapter'], size=9)
            ax.legend(numpoints=1)
            plt.savefig("Cluster-%d" % name)
            plt.close()

    def run_all(self):
        self.generate_chapter_and_text_lists()
        self.perform_tfidf()
        self.perform_clustering(self.NUM_CLUSTERS)
        self.print_clusters()
        self.generate_cluster_plot_frame()
        self.plot_all_clusters()
        self.plot_all_subplots()
        self.plot_sequential_graphs()

myThing = HarryPotter()
myThing.run_all()

