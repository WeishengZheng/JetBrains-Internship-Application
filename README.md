# JetBrains-Internship-Application
## Task #1

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).
The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.


## Word2vec implementation
Sources:

[Efficient Estimation of Word Representations in Vector Space
Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean](https://arxiv.org/pdf/1301.3781)

[Word2vec with gensim](https://www.geeksforgeeks.org/nlp/word2vec-with-gensim/)

### Implementation path
- Data processing
    - Tokenization
    - Lowercasting
    - Removing Stopwords
- CBOW implementation


