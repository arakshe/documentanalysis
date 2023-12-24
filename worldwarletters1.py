"""
filename: worldwarletters.py
description: An extensible reusable library for text analysis and comparison
"""
import pandas as pd
from collections import defaultdict, Counter
import random as rnd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string
from matplotlib.sankey import Sankey
from textblob import TextBlob
import nltk
import numpy as np


class WorldWarLetters:
    """A class for analyzing and visualizing text data from World War I letters.
        Attributes:
            data (defaultdict): A dictionary to hold various text statistics.
            CUSTOM_STOPWORDS (set): A set of stopwords for text processing.
        """
    CUSTOM_STOPWORDS = {"it","are","for", "be", "to", "me", "the", "this", "was", "am", "but" , "one", "its", "a", "i", "my", "as", "on", "at", "that", "in", "your", "you", "too",
                        "with", "of", "and", "is", "about", "there", "seems", "again"}

    def __init__(self):
        """Initialize the WorldWarLetters class with empty data structures and load custom stopwords."""

        self.data = defaultdict(dict)

    def _save_results(self, label, results):
        for k, v in results.items():
            self.data[k][label] = v

    @staticmethod
    def _default_parser(filename):
        """Default parser for text files to extract word counts and filter out stopwords and short words.
        Args:
            filename (str): The path to the text file to be parsed.
        Returns:
            dict: A dictionary with word counts and the total number of words.
        """

        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()

        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        words = text.split()

        # Filter out stopwords
        filtered_words = [word for word in words if word not in WorldWarLetters.CUSTOM_STOPWORDS and len(word) > 4]

        wordcount = Counter(filtered_words)
        numwords = len(filtered_words)

        results = {
            'wordcount': wordcount,
            'numwords': numwords
        }
        return results

    def load_text(self, filename, label=None, parser=None):
        """ Registers a text document with the framework
        Extracts and stores data to be used in later visualizations. """

        if parser is None:
            results = WorldWarLetters._default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        self._save_results(label, results)

    def compare_num_words(self):
        """Generate a bar chart comparing the number of words in each registered text document."""

        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.show()

    def load_text(self, filename, label=None, parser=None):
        """ Registers a text document with the framework
        Extracts and stores data to be used in later visualizations. """

        if parser is None:
            results = WorldWarLetters._default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        self._save_results(label, results)

    def compare_num_words(self):
        """ A DEMONSTRATION OF A CUSTOM VISUALIZATION
        A trivially simple barchart comparing number
        of words in each registered text file. """

        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.show()


    def wordcount_sankey(self, k=15):
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                     "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                     "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
                     "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                     "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
                     "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                     "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
                     "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
                     "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                     "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                     "will", "just", "don", "should", "now"]
        word_counts = Counter()
        for wordcount in self.data['wordcount'].values():
            filtered_wordcount = {word: count for word, count in wordcount.items() if word not in stopwords}
            word_counts += Counter(filtered_wordcount)
        top_k_words = word_counts.most_common(k)
        file_list = []
        word_list = []
        count_list = []

        # Assign a unique color index to each word in top_k_words
        word_color_map = {word: i for i, word in enumerate(top_k_words)}

        for i, (word, _) in enumerate(top_k_words):
            for file, wordcount in self.data['wordcount'].items():
                count = wordcount.get(word, 0)
                if count > 0:
                    file_list.append(file)
                    word_list.append(word)
                    count_list.append(count)

        df = pd.DataFrame({'File': file_list, 'Word': word_list, 'Count': count_list})
        colors = sns.color_palette("hls", n_colors=k).as_hex()
        unique_words = df['Word'].unique()
        color_mapping = dict(zip(unique_words, colors))
        # Add a new 'Color' column based on the mapping
        df['Color'] = df['Word'].map(color_mapping)
        src, targ, val = 'File', 'Word', 'Count'
        labels = list(set(df[src].unique()) | set(df[targ].unique()))
        lc_map = {label: i for i, label in enumerate(labels)}
        df[src] = df[src].map(lc_map)
        df[targ] = df[targ].map(lc_map)
        link_color = [word_color_map.get(word, df['Color'][index]) for index, word in enumerate(df['Word'])]
        link = {'source': df[src], 'target': df[targ], 'value': df[val], 'color': link_color}
        node = {'label': labels, 'pad': 15, 'thickness': 20, 'color': 'lightgrey'}
        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
        fig.update_layout(title_text="Sankey Diagram of " + str(k) + " Most Common Words in World War 1 Letters", font_size=20)
        fig.show()


    def generate_wordclouds(self, subplot_dims=(2, 5)):
        """Create a series of word clouds from the text data.
                Args:
                    subplot_dims (tuple, optional): The dimensions for the subplot grid. Defaults to (2, 5).
                """
        num_files = len(self.data['wordcount'])
        if num_files < 2 or num_files > 10:
            raise ValueError("Number of files must be between 2 and 10.")

        rows, cols = subplot_dims
        if rows * cols < num_files:
            raise ValueError("Subplot dimensions are too small for the number of files.")

        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        axes = axes.flatten()

        for i, (label, wordcount) in enumerate(self.data['wordcount'].items()):
            wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(wordcount)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(label)

        for j in range(i + 1, rows * cols):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def vis_sentiment(self):
        """ Analyzes sentiment of loaded texts and creates a comparative visualization. """
        polarities = []
        subjectivities = []
        labels = []

        for label, text_data in self.data['wordcount'].items():
            text = ' '.join(text_data.elements())  # Reconstruct text from word count
            blob = TextBlob(text)
            polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity

            polarities.append(polarity)
            subjectivities.append(subjectivity)
            labels.append(label)

        # Add a small random noise to polarities and subjectivities
        noise = np.random.normal(0, 0.01, len(polarities))
        polarities = [p + n for p, n in zip(polarities, noise)]
        subjectivities = [s + n for s, n in zip(subjectivities, noise)]

        # Plotting
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.scatter(polarities[i], subjectivities[i], label=label)

        plt.xlabel('Polarity')
        plt.ylabel('Subjectivity')
        plt.title('Comparative Sentiment Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()


worldwar = WorldWarLetters()

file1 = {
    "filename": "letter1.txt",
}

file2 = {
    "filename": "letter2.txt",
}

file3 = {
    "filename": "letter3.txt"
}
file4 = {
    "filename": "letter4.txt"
}
file5 = {
    "filename": "letter5.txt"
}
worldwar.load_text(file1['filename'], label='Letter 1')
worldwar.load_text(file2['filename'], label='Letter 2')
worldwar.load_text(file3['filename'], label='Letter 3')
worldwar.load_text(file4['filename'], label='Letter 4')
worldwar.load_text(file5['filename'], label='Letter 5')

worldwar.generate_wordclouds()
worldwar.wordcount_sankey()
worldwar.vis_sentiment()
