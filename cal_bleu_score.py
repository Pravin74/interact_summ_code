import os
import nltk

def cal_bleu_score():
    hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
    reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    print(BLEUscore)

def main():
    cal_bleu_score()    

if __name__ == "__main__":
    main()

