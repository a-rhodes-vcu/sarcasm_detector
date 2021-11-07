import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow as tf
import numpy as np
import head_line_web_scraper


class BuildSarcasmData:
    """
    BuildSarcasmData is a sentiment analyst which that takes headlines stored in a json file and creates a training set.
    The training set is used to create a model that can predict if web scraped headlines
    are sarcastic or not. json file is from kaggle.
    """

    def __init__(self):

        # lists used to store json contents
        self.sentences = []
        self.labels = []
        self.urls = []

        # lists used to store training and testing data
        self.training_sentences = []
        self.training_labels = []
        self.testing_sentences = []
        self.testing_labels = []

        # initialize tokenizer variables
        self.vocab_size = 20000
        self.tokenizer = None

    def process_data(self):
        """
        Process the data and to prep for training and testing data
        """
        with open("Sarcasm_Headlines_Dataset_v2.json", "r") as f:
            data = json.load(f)

        for value in data:
            self.sentences.append(value["headline"])
            self.labels.append(value["is_sarcastic"])
            self.urls.append((value["article_link"]))

        # larger training size results in a more accurate prediction
        training_size = 90000
        # training lists
        self.training_sentences = self.sentences[0:training_size]
        self.training_labels = self.labels[0:training_size]

        # testing lists
        self.testing_sentences = self.sentences[training_size:]
        self.testing_labels = self.labels[training_size:]

        # create tokenizer - separate headlines into words
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        # create vocab index based on word frequency
        self.tokenizer.fit_on_texts(self.sentences)

    def create_model(self):
        """
        Take processed data and create training and testing sequences
        """
        embedding_dim = 16
        max_length = 200
        trunc_type = 'post'
        padding_type = 'post'
        training_sequences = self.tokenizer.texts_to_sequences(self.training_sentences)
        training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                        padding=padding_type, truncating=trunc_type)

        testing_sequences = self.tokenizer.texts_to_sequences(self.testing_sentences)
        testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                       padding=padding_type, truncating=trunc_type)

        training_padded = np.array(training_padded)
        training_labels = np.array(self.training_labels)
        testing_padded = np.array(testing_padded)
        testing_labels = np.array(self.testing_labels)

        # Create model - 3 layers
        # used the sequential model which is a stack of layers feeding linearly from one to the next
        # the output of one layer is input to next layer
        model = tf.keras.Sequential([
            # embedding layer is first layer of model where each word is represented by a dense vector
            tf.keras.layers.Embedding(self.vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            # dense layer with 50 neurons
            tf.keras.layers.Dense(50, activation='relu'),
            # drop out 50% of neurons to prevent over fitting and can lower variability of neural network
            tf.keras.layers.Dropout(0.5),
            # output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # compile model using SGD optimizer and cross-entropy.
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # number of times training set is passed through model
        num_epochs = 30
        # train model with inputs
        model.fit(training_padded, training_labels, epochs=num_epochs,
                  validation_data=(testing_padded, testing_labels), verbose=2)

        # web scrape headlines
        news_head_lines = (head_line_web_scraper.get_news_headlines()[0:4])
        sarcastic_news_head_lines = (head_line_web_scraper.get_sarcastic_headlines()[0:4])
        news_head_lines.extend(sarcastic_news_head_lines)

        # process web headlines
        sequences = self.tokenizer.texts_to_sequences(news_head_lines)
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        headline_list = []
        prediction_list = []

        # receive prediction from model with web headlines
        for j, i in zip(model.predict(padded), news_head_lines):
            headline_list.append(i)
            prediction_list.append(f"{'{0:.2f}'.format(float(j[0]) * 100)}% sarcastic")

        # format output
        d = {'col2': headline_list, 'col3': prediction_list}
        df = pd.DataFrame(data=d)
        df.style.set_properties(**{'text-align': 'right'})
        pd.set_option('display.max_colwidth', 1000)
        print(df)


td = BuildSarcasmData()
td.process_data()
td.create_model()
