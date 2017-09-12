import string
import os
#import nltk.tokenize as tokenizer
import datetime
from nltk.tokenize import TweetTokenizer

languages = ['fr','es','ru','ar','zh']
sourcePath = "C:\\NLP"
tknzr = TweetTokenizer()
# print("hello")
# s = "Good hel'lo muffins cost can't $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
# myTokenizer = tokenizer.RegexpTokenizer(r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S')
# aa= [word.strip(string.punctuation) for word in s.split(" ")]
# print(aa)
for lang in languages:
    print("***Starting Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(sourcePath,lang,"en.txt"),'r',encoding='utf-8')as readFile:
        with open(os.path.join(sourcePath,lang,"en_tweetsTokenizer.txt"),'w', encoding= 'utf-8') as writeFile:

            # data = readFile.read()
            # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            # tokenizer.tokenize(data)

            for line in readFile:
                #writeFile.write(" ".join([word.strip(string.punctuation) for word in line.split(" ")]))
                writeFile.write(" ".join(tknzr.tokenize(line)) + '\n')

    print("***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
