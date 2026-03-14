import Word2vec.src.preprocess as preprocess
import src.indexer as indexer
# python3 -m pytest test/test.py

def test_tokenizer():
    text = "HeLlo world! given hue It's 934 fler)) fe$ is in fer543 the"
    
    expected = ['hello', 'given', 'hue', "it's", '#']

    result = preprocess.tokenizer(text)

    assert result == expected

def test_round_trip_indexer():
    basic_indexer = indexer.Indexer()

    basic_indexer.index_loader("/workspaces/JetBrains-Internship-Application/Word2vec/Data/tokens_index")
    
    test_words = ["hello", "dvorak", "animal", "coffee", "sun"]

    for word in test_words:
        index = basic_indexer.getIndex(word)
        assert basic_indexer.getToken(index) == word