import src.process as process

def test_tokenizer():
    text = "HeLlo world! given hue It's 934 fler)) fe$ fer543"
    
    expected = ['hello', 'given', 'hue', "it's", '#']

    result = process.tokenizer(text)

    assert result == expected