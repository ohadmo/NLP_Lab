# go over the files, for every language file, group all original lines together. divide to chunks
# find all pronouns in the corpus
# count pronouns frequencies.
# data points = chunks, dimension = number of pronouns in corpus. values = frequencies. label = {T, O}

#from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np
import sys
import datetime
import pickle
import os.path
import time
from multiprocessing import Process, Queue, current_process
#sys.path.append('D:\ohadm\Downloads\libsvm-3.22\python')
#import svmutil
import operator
from enum import Enum
import argparse

CHUNK_SIZE = 2000

class SearchWordsOrPOS(Enum):
    FIND_WORDS = 0
    FIND_POS = 1


def readWordsFromFile(path):
    print('in read words from file: ' + path)
    wordsSet = set()
    with open(path,'r', encoding='utf-8') as file:
        for line in file:
            ln = line.strip().lower().split(',')
            for word in ln:
                word = word.strip()
                if not word == '':
                    wordsSet.add(word)
    return wordsSet

def wordsInCorpus(path, words_set):
    print('words in corpus: ' + path)
    valid_words = set()
    with open(path, 'r', encoding='cp1252') as eng_file:
        for line in eng_file:
            if len(valid_words) == len(words_set):
                return valid_words
            tokens = [x.strip().lower().split('_')[0] for x in line.split(' ')]
            for token in tokens:
                if token in words_set:
                    valid_words.add(token)
    return valid_words

def POSinCorpus(path, ngrams):
    all_POS = dict()
    with open(path,'r') as file:
        for line in file:
            try:
                pos_line = [x.strip().split('_')[1] for x in line.split(' ')]
                if len(pos_line) < ngrams:
                    continue
                for i in range(len(pos_line) - ngrams + 1):
                    pos = ""
                    for k in range(ngrams):
                        pos += pos_line[i+k] + "_"
                    pos = pos[:-1]
                    if pos not in all_POS:
                        all_POS[pos] = 1
                    else:
                        all_POS[pos] += 1
            except:
                print("Exception POS line is:")
                print(line)
    #valid_POS = sorted(all_POS.items(), key=operator.itemgetter(1), reverse=True) # Decreasing Order
    #temp = [x[0] for x in valid_POS]
    #temp = temp[:400]
    #valid_POS = set(temp)
    return dict(all_POS)

def ChunkToWordsCounters(chunk, cNouns, idx):
    # tokens = [x.strip().lower().split('_')[idx.value] for x in chunk]
    pronouns_count = {x: 0 for x in cNouns}
    for token in chunk:
        if token in cNouns:
            pronouns_count[token] += 1
    return [((x / len(chunk))*(2000/len(chunk))) for x in  pronouns_count.values()]


def divide_to_chnuks(language_file, label_file, lang, corpusNouns, search_enum):
    print("start divide to chunks")
    original_chunks = []
    original_chunk = []
    translated_chunks = []
    translated_chunk = []
    with open(language_file, 'r', encoding='cp1252') as dfile, open(label_file, 'r', encoding='utf-8') as lfile:
        for line, label in zip(dfile, lfile):
            tokens = line.split(' ')
            tokens = [x.strip().lower().split('_')[search_enum.value] for x in tokens]
            if search_enum == SearchWordsOrPOS.FIND_POS:
                t = []
                for i in range(len(tokens) - 3 + 1):
                    pos = ""
                    for k in range(3):
                        pos += tokens[i + k] + "_"
                    pos = pos[:-1]
                    t.append(pos.upper())
                tokens = t
            if label.strip() == lang:
                original_chunk += tokens
                if len(original_chunk) > CHUNK_SIZE:
                    original_chunks.append(ChunkToWordsCounters(original_chunk, corpusNouns, search_enum))
                    original_chunk = []
            else:
                translated_chunk += tokens
                if len(translated_chunk) > CHUNK_SIZE:
                    translated_chunks.append(ChunkToWordsCounters(translated_chunk, corpusNouns, search_enum))
                    translated_chunk = []
    original_chunks.append(ChunkToWordsCounters(original_chunk, corpusNouns, search_enum))
    translated_chunks.append(ChunkToWordsCounters(translated_chunk, corpusNouns, search_enum))
    return original_chunks, translated_chunks


def incrementCount(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 1

if __name__ == '__main__':
    print( "***Starting Time*** "  + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    parser = argparse.ArgumentParser(description='This script runs an ML trained model according to specified features')
    parser.add_argument('features', help='Search for words or post in corpus POSBigrams POSTrigrams FunctionWords Pronouns',
                        choices=['POSBigrams', 'POSTrigrams', 'FunctionWords', 'Pronouns'])
    parser.add_argument('--pronouns_path', help='Specify the pronouns file path', required=True)
    parser.add_argument('--funcwords_path', help='Specify the function words file path', required=True)
    parser.add_argument('--dicts_path', help="dicts folder path", required=True)
    parser.add_argument('--languages', help="languages to work on from dict",required=True)
    args = parser.parse_args()
    print(args)


    if args.features == "Pronouns" or args.features == "FunctionWords":
        if args.features == "Pronouns":
            file_words = readWordsFromFile(args.pronouns_path)
        elif args.features == "FunctionWords":
            valid_words_in_corpus = set()
            file_words = readWordsFromFile(args.funcwords_path)
            for lang in args.languages.split(" "):
                function_words_pickle_path = os.path.join(args.dicts_path, lang, 'function_words.p')
                if os.path.isfile(function_words_pickle_path):
                    print("*In lang " + lang + " loading function words from pickle file")
                    valid_words_in_corpus = valid_words_in_corpus | pickle.load(open (function_words_pickle_path, 'rb'))
                else:
                    print("*In lang " + lang + " calling wordsInCorpus and saving function words to pickle")
                    valid_words_in_one_corpus = wordsInCorpus(os.path.join(args.dicts_path, lang, 'en_tagged_tweetTokenizer.txt'), file_words)
                    pickle.dump(valid_words_in_one_corpus,  open(function_words_pickle_path, 'wb') )
                    valid_words_in_corpus =  valid_words_in_corpus | valid_words_in_one_corpus
            for lang in args.languages.split(" "):
                dataOrg_file =  os.path.join(args.dicts_path, lang, 'dataOriginal_fw.p')
                labelTrans_file = os.path.join(args.dicts_path, lang, 'dataTranslated_fw.p')
                if os.path.isfile(dataOrg_file) and os.path.isfile(labelTrans_file):
                    print("*In lang " + lang + " loading data.p and label.p from pickle file")
                    data = pickle.load(open(dataOrg_file, 'rb'))
                    label = pickle.load(open(labelTrans_file, 'rb'))
                else:
                    print("*In lang " + lang + " calling devide to chunks and dumping data.p and label.p ")
                    o, t = divide_to_chnuks(os.path.join(args.dicts_path, lang , 'en_tagged_tweetTokenizer.txt'),
                                            os.path.join(args.dicts_path, lang, lang + "-en.txt"),
                                            'en', valid_words_in_corpus, SearchWordsOrPOS.FIND_WORDS)
                    #data, label = combineSamplesNormalize(o, t)
                    print("no. of sampels in original: " + str(len(o)))
                    print("no. of sampels in translated: " + str(len(t)))
                    pickle.dump(o, open(dataOrg_file, 'wb'))
                    pickle.dump(t, open(labelTrans_file, 'wb'))

    if "POS" in args.features:
        valid_pos_in_corpus = dict()
        if args.features == "POSTrigrams":
            for lang in args.languages.split(" "):
                posTrigram_pickle_path = os.path.join(args.dicts_path, lang, '3pos.p')
                if os.path.isfile(posTrigram_pickle_path):
                    print("*In lang " + lang + " loading pos 3-gram words from pickle file")
                    valid_words_in_one = pickle.load(open(posTrigram_pickle_path, 'rb'))
                    for k,v in valid_words_in_one.items():
                        if k in valid_pos_in_corpus:
                            valid_pos_in_corpus[k] += v
                        else:
                            valid_pos_in_corpus[k] = v
                else:
                    print("*In lang " + lang + " calling POSinCorpus and saving 3posgram to pickle")
                    valid_words_in_one = POSinCorpus(os.path.join(args.dicts_path, lang,"en_tagged_tweetTokenizer.txt"), 3)
                    pickle.dump(valid_words_in_one, open(posTrigram_pickle_path, 'wb'))
                    for k,v in valid_words_in_one.items():
                        if k in valid_pos_in_corpus:
                            valid_pos_in_corpus[k] += v
                        else:
                            valid_pos_in_corpus[k] = v
            valid_POS = sorted(valid_pos_in_corpus.items(), key=operator.itemgetter(1), reverse=True) # Decreasing Order
            temp = [x[0] for x in valid_POS][:400]
            valid_POS = set(temp)
            for lang in args.languages.split(" "):
                dataOrg_file = os.path.join(args.dicts_path, lang, 'dataOriginal_pos.p')
                dataTrans_file = os.path.join(args.dicts_path, lang, 'dataTranslated_pos.p')
                if os.path.isfile(dataOrg_file) and os.path.isfile(dataTrans_file):
                    print("*In lang " + lang + " loading data_pos.p and label_pos.p from pickle file")
                    data = pickle.load(open(dataOrg_file, 'rb'))
                    label = pickle.load(open(dataTrans_file, 'rb'))
                else:
                    print("*In lang " + lang + " calling devide to chunks and dumping data_pos.p and label_pos.p ")
                    o, t = divide_to_chnuks(os.path.join(args.dicts_path, lang , 'en_tagged_tweetTokenizer.txt'),
                                            os.path.join(args.dicts_path, lang, lang + "-en.txt"),
                                            'en', valid_POS, SearchWordsOrPOS.FIND_POS)
                    #data, label = combineSamplesNormalize(o, t)
                    print("no. of sampels in original: " + str(len(o)))
                    print("no. of sampels in translated: " + str(len(t)))
                    pickle.dump(np.array(o), open(dataOrg_file, 'wb'))
                    pickle.dump(np.array(t), open(dataTrans_file, 'wb'))

        elif args.features == "POSBigrams":
            pass
            #valid_words_in_corpus = POSinCorpus(args.english_path, 2)
            #o, t = divide_to_chnuks(args.english_path, args.labels_path, 'en', valid_words_in_corpus, SearchWordsOrPOS.FIND_POS)

    print("***Starting MultiProcesses*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
