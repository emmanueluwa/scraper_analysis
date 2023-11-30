import sys 
from review_model import ReviewModel
import subprocess
import json

# review_url = sys.argv[2]

# subprocess.run(["node", "singleReviewParser.js", ])

# read .txt file created by nodejs webscrape
with open("best_review_temp.txt") as f:
    lines = f.readlines()
    review_text = "\n".join(lines)

f = open("review_corpus.json")
#return json object as dict
review_dataset = json.load(f)

review_model = ReviewModel(review_dataset=review_dataset)

review_model.load_model()

results = review_model.rank(review_text, top_n=3)
print(results)