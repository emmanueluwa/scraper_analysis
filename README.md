# scraper x scraper

- This tool will find similar reviews(by url) based on review passed to script

ML method

- crawl through much more reviews and scrape the content

- create a script to scrape the ideal business niche with reviews?

- create embeddings for entire review corpus

  - pre-train a word2vec (doc2vec) model
    [https://en.wikipedia.org/wiki/Word2vec]

  - all information will be pre-crawled

    - new reviews will not be fetched on inference
    - fetch from DB for each search

- the process used to find similar reviews

  - cosine similarity [https://en.wikipedia.org/wiki/Cosine_similarity]

- DB

  - json file for now
  - switch to mongoDB? Use best DB for unstructured data

  [
  {
  url: "https://",
  name: "...",
  embedding: []
  }
  ]
