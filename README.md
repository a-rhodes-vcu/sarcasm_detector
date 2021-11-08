# Sarcasm Dectector
This project is essentially a sentiment analyzer that detects whether a headline is sarcastic or not. 

## Motivation
I think it's fascinating that the feeling of a sentence can be extracted and I wanted to create a sentiment analyzer that could predict the tone of a phrase.

## Finished Product
To run in your local environment you just need to install the packages found in the [requirements.txt](https://github.com/a-rhodes-vcu/sarcasm_detector/blob/main/requirements.txt)

## Code walkthrough
Programs used:
<br>
[sarcasm_analyzer.py](https://github.com/a-rhodes-vcu/sarcasm_detector/blob/main/sarcasm_analyzer.py)
<br>
[Sarcasm_Headlines_Dataset_v2.json](https://github.com/a-rhodes-vcu/sarcasm_detector/blob/main/Sarcasm_Headlines_Dataset_v2.json)
<br>
[head_line_web_scraper.py](https://github.com/a-rhodes-vcu/sarcasm_detector/blob/main/head_line_web_scraper.py)
<br>

The [Sarcasm_Headlines_Dataset_v2.json](https://github.com/a-rhodes-vcu/sarcasm_detector/blob/main/Sarcasm_Headlines_Dataset_v2.json) file contains key/value pairs of a boolean value for sarcasism or not, the headline and the source. 
```
[{"is_sarcastic": 1, "headline": "thirtysomething scientists unveil doomsday clock of hair loss", "article_link": "https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205"},
{"is_sarcastic": 0, "headline": "dem rep. totally nails why congress is falling short on gender, racial equality", "article_link": "https://www.huffingtonpost.com/entry/donna-edwards-inequality_us_57455f7fe4b055bb1170b207"},
```
In [sarcasm_analyzer.py](https://github.com/a-rhodes-vcu/sarcasm_detector/blob/main/sarcasm_analyzer.py) the json file is iterated over and stored into lists

```python
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
```
Now it's time to tokenize, which is taking a sentence and breaking it into words and then creating an index of word frequencies. 
```python
 
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
```
create_model creates the inputs for the neural network which are lists of testing and training data. texts_to_sequences() creates a list of lists where each word is represented by a number and pad_sequences() is used to pad the sequences so each one has identical lengths.
```python
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
```
Finally to the neural network! The neural network has three layers, first layer is the embedding layer, second layer has 50 neurons, followed by the output layer. 50% of the neurons are dropped out to prevent overfitting. The model is then compiled and fit with the training and testing data. 
```python
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
```
Now for the fun part, headlines are scraped from CBS and The Onion, processed and then put through the model to determine a prediction. The scraped headline and prediction is outputted.
```python
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
```
Output is a pandas dataframe that contains four CBS headlines and four The Onion headlines. What I think is cool is that each day the headlines and predictions will be different due to updates in the news cycle. Unfortunately the model did misidentify a CBS news headline as being sarcastic and an Onion headline as not sarcastic. There are many possibilities as to why the model did not accurately identify a sarcastic/not sarcastic headline, some of them could be due to not enough layers in the neural network or that the dataset only has headlines from two sources.
```
0                                 Bomb threats called into multiple Ivy League universities   94.50% sarcastic
1                         What we know about the victims of the Astroworld Festival tragedy    0.00% sarcastic
2                  Gottlieb predicts "broad immunity" among children as more get vaccinated    2.71% sarcastic
3                               Kenyans Albert Korir and Peres Jepchirchir win NYC Marathon    0.00% sarcastic
4             Breakthrough Renewable Energy Technology Enables Humans To Burn Wind For Fuel    0.13% sarcastic
5  Construction Finally Complete On Canal Connecting Chemical Runoff With Mississippi River  100.00% sarcastic
6    Report: Catapult Industry Won’t Survive Another Year Without Medieval War Breaking Out  100.00% sarcastic
7   Study Shows Tapping Cheek With Pointer Finger Still Number One Way To Get A Little Kiss  100.00% sarcastic
```

## Improvements
1. Update the .json file to contain news headlines from multiple different news sources rather than just two
2. Fine tuning of the neural network or how the data is processed could result in a more accurate prediction.
3. It would be interesting to scrape headlines from Weekly World News (if you know, you know) and determine if the model could predict Bat Boy, etc headlines as being sarcastic.

## Tech used
[Python](https://www.python.org/) 3.7


<h6>If you made it to the end, thanks for reading! Feel free to reach out for questions or suggesstions! This was such a fun project to work on and I look forward to more future NLP/neural network projects.</h6>
