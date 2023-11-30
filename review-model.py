#simple word2vec model
import numpy as np 
import nltk
import time
import json


from gensim.models import doc2vec
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec



#tokenising
nltk.download("punkt")


# as the model is trained the call back gives async event based feedback
class callback(CallbackAny2Vec):
    #Callback to print loss after each epoch

    def __init__(self):
        self.epoch = 0
        self.epoch_start_time = 0

    def on_epoch_begin(self, model):
        # model.running_training_loss=0
        print("Epoch #{} start".format(self.epoch))
        self.epoch_start_time = time.time()

    def on_epoch_end(self, model):
        #computing loss does not work?
        # loss = model.get_latest_training_loss()
        # print("Loss: {}".format(loss))
        epoch_time = time.time() - self.epoch_start_time
        print("Time of execution:", str(epoch_time) + "s")
        self.epoch += 1

class ReviewModel:
    def __init__(self, review_dataset):
        #list of large strings for each review
        self.review_dataset = review_dataset
        self.model = doc2vec.Doc2Vec(vector_size=32, min_count=1, window=20, workers=2, epochs=2, compute_loss=True)

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
        print("Training model")
        self.model.train(self.training_data, total_examples=self.model.corpus_count, epochs=self.model.epochs, compute_loss=True, callbacks=[callback()])
        print("Succesfully trained the model")


    def generate_embeddings_for_dataset(self):
        print("Generating embeddings for dataset")
        for i, review in enumerate(self.review_dataset):
            #vectorised representation of the data
            embedding = self.predict(review["reviewText"])
            self.review_dataset[i]["embedding"] = embedding


    def save_datatset(self):
        print("Saving dataset")
        with open("review_corpus.json", "w+") as f:
            f.write(json.dumps(self.review_dataset))


    #infer vector -> make prediction
    def predict(self, review):
        vector = self.model.infer_vector(simple_preprocess(review)).tolist()
        return vector

if __name__ == "__main__":

    #open json file
    f = open("review_corpus.json")
    #return json object as dict
    review_dataset = json.load(f)

    review = ReviewModel(review_dataset)
    review.train()

    review.generate_embeddings_for_dataset()
    review.save_datatset()
    
    # _input = "This is a service"
    # print(_input)
    # prediction = review.predict(_input)
    # print(prediction)



        