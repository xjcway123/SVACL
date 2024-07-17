
buckets = 250499

def biGramHash(code_sequence, t, buckets = buckets):
    t1 = code_sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(code_sequence, t, buckets = buckets):
    t1 = code_sequence[t - 1] if t - 1 >= 0 else 0
    t2 = code_sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

def getBiGram(code_sequence):
    """
        code_sequence: 128-length token sequence
        get 2-gram
    """
    assert len(code_sequence) == 128
    bigram = []
    for i in range(128):
        bigram.append(biGramHash(code_sequence, i))
    return bigram

def gettriGram(code_sequence):
    """
        code_sequence: 128-length token sequence
        get 3-gram
    """
    assert len(code_sequence) == 128
    trigram = []
    for i in range(128):
        trigram.append(triGramHash(code_sequence, i))
    return trigram