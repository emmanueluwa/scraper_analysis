#simple word2vec model
import numpy as np 
import nltk

import json


from gensim.models import doc2vec
from gensim.utils import simple_preprocess


#tokenising
nltk.download("punkt")

class ReviewModel:
    def __init__(self, review_dataset):
        #list of large strings for each review
        self.review_dataset = review_dataset
        self.model = doc2vec.Doc2Vec(vector_size=2, min_count=1, window=3, workers=2, epochs=2)

        self.training_data = self.process_training_data()

        self.model.build_vocab(self.training_data)
        
        words = list(self.model.wv.key_to_index.keys())
        #constraining vocab size with large corpus of data, increases chance of rep of word being more powerful?
        print("Vocab size:", len(words))

    def process_training_data(self):
        #list of tagged reviews for each doc, main difference between word2vec and doc2vec
        training_data = [doc2vec.TaggedDocument(simple_preprocess(review["reviewText"]), [index]) for index, review in enumerate(review_dataset)]
        return training_data


    def train(self):
        self.model.train(self.training_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        print("succesfully trained the model")

    def generate_embeddings_for_database(self):
        pass

    #infer vector -> make prediction
    def predict(self, review):
        vector = self.model.infer_vector(review.split(" ")).tolist()
        return vector

if __name__ == "__main__":

    #open json file
    f = open("review_corpus.json")
    #return json object as dict
    review_dataset = json.load(f)

    review = ReviewModel(review_dataset)
    # review.train()

    # _input = "This is a service"
    # print(_input)
    # prediction = review.predict(_input)
    # print(prediction)