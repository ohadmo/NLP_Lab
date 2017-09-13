import os
import datetime
from nltk.tokenize import TweetTokenizer
import argparse

tknzr = TweetTokenizer()
parser = argparse.ArgumentParser(description='This script runs the tokenizer over en.txt in the specified languages'
                                             ' folder, then writes the output to en_tweetsTokenizer.txt')
parser.add_argument('--working_dir_path',
                    help='Specify the working file path. Note:must contain folders named as the languages provided',
                    required=True)
parser.add_argument('--languages',
                    help='languages to apply this script on separated by a blank space',
                    required=True)
args = parser.parse_args()

for lang in args.languages.split(" "):
    print(lang + "***Starting Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(args.working_dir_path, lang, "en.txt"),'r',encoding='utf-8')as readFile:
        with open(os.path.join(args.working_dir_path, lang, "en_tweetsTokenizer.txt"),'w', encoding= 'utf-8') as writeFile:
            for line in readFile:
                writeFile.write(" ".join(tknzr.tokenize(line)) + '\n')
    print(lang + "***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))