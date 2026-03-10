import src.process as process

def test_tokenizer():
    text = "HeLlo world! given hue It's 934 fler)) fe$ is in fer543 the"
    
    expected = ['hello', 'given', 'hue', "it's", '#']

    result = process.tokenizer(text)

    assert result == expected