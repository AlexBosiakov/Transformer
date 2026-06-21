import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="../data/spm_corpus.txt",
    model_prefix="spm_20000",
    vocab_size=20000,  # Размер словаря
    model_type="bpe",  # Тип токенизации: Byte Pair Encoding
    max_sentence_length=2048,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

