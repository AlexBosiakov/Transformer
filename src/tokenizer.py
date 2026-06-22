import sentencepiece as spm

class SPMTokenizer:
    def __init__(self, model_path="../spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.cls_id = self.sp.piece_to_id("<cls>")

    def encode(self, text, max_len=256):
        # Превращает строку text в список id токенов длины max_len
        ids = self.sp.encode(text, out_type=int)

        ids = [self.cls_id] + ids

        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))  # PAD=0

        return ids

    def decode(self, ids):
        # Из id токенов в текст
        return self.sp.decode(ids)
