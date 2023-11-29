#simple word2vec model
import numpy as np 
import nltk


from gensim.models import doc2vec

#tokenising
nltk.download("punkt")

class ReviewModel:
    def __init__(self, reviewList):
        #list of large strings for each review
        self.reviewList = reviewList
        self.model = doc2vec.Doc2Vec(vector_size=2, min_count=1, window=3, workers=2, epochs=2)

        #list of tagged reviews for each doc, main difference between word2vec and doc2vec
        self.training_data = [doc2vec.TaggedDocument(review, [index]) for index, review  in enumerate(reviewList)]

        self.model.build_vocab(self.training_data)
        
        words = list(self.model.wv.key_to_index.keys())
        #constraining vocab size with large corpus of data, increases chance of rep of word being more powerful?
        print("Vocab size:", len(words))

    def train(self):
        self.model.train(self.training_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        print("succesfully trained the model")

    #infer vector -> make prediction
    def predict(self, review):
        vector = self.model.infer_vector(review.split(" ")).tolist()
        return vector

if __name__ == "__main__":
    review = ReviewModel(["This is a review for a electric charger company", "This is another review for a top electric charger company", "This is another review for a electric car specialist garage", "This is a review for fracture therapy", "This is not a service at all"])
    review.train()

    # _input = "This is a service"
    # print(_input)
    # prediction = review.predict(_input)
    # print(prediction)