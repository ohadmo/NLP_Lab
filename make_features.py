#go over the files, for every language file, group all original lines together. divide to chunks
#find all pronouns in the corpus
#count pronouns frequencies.
#data points = chunks, dimension = number of pronouns in corpus. values = frequencies. label = {T, O}

#from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np
import sys
import datetime
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
    valid_words_in_corpus = set()
    with open(path, 'r') as eng_file:
        for line in eng_file:
            tokens = [x.strip().lower().split('_')[0] for x in line.split(' ')]
            for token in tokens:
                if token in words_set:
                    valid_words_in_corpus.add(token)
    return valid_words_in_corpus

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
    valid_POS = sorted(all_POS.items(), key=operator.itemgetter(1), reverse=True) # Decreasing Order
    temp = [x[0] for x in valid_POS]
    temp = temp[:400]
    valid_POS = set(temp)
    return valid_POS

def ChunkToWordsCounters(chunk, cNouns, idx):
    tokens = [x.strip().lower().split('_')[idx.value] for x in chunk]
    pronouns_count = {x: 0 for x in cNouns}
    for token in tokens:
        if token in cNouns:
            pronouns_count[token] += 1
    return [x for x in pronouns_count.values()]

def divide_to_chnuks(language_file, label_file, lang, corpusNouns, search_enum):
    original_chunks = []
    original_chunk = []
    translated_chunks = []
    translated_chunk = []
    with open(language_file, 'r') as dfile, open(label_file, 'r', encoding='utf-8') as lfile:
        for line, label in zip(dfile, lfile):
            tokens = line.split(' ')
            tokens = [x.strip() for x in tokens]
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

def combineSamplesNormalize(original_samples, translated_samples):
    np.random.shuffle(original_samples)
    np.random.shuffle(translated_samples)
    print("no. of sampels in original: " + str(len(original_samples)))
    print("no. of sampels in translated: " + str(len(translated_samples)))
    retData = []
    retLabels = []
    for i in range(len(translated_samples)):
        retData.append(original_samples[i])
        retLabels.append(1)
    for i in range(len(translated_samples)):
        retData.append(translated_samples[i])
        retLabels.append(0)
    '''
    mean = np.sum(retData, axis=0) / len(retData)
    temp = -1* np.tile(mean,(len(retData),1))
    variance = (np.array(retData) + temp)
    variance = variance**2
    variance = np.sum(variance, axis=0) / len(retData)
    variance = np.sqrt(variance)
    retData = (retData-mean)/variance
    '''
    _min = np.min(retData, axis=0)
    _max = np.max(retData, axis=0)
    retData = (retData - _min) / (_max - _min)
    '''
    print("Infinite indices:")
    for i in range(len(retData)):
        if np.isfinite(retData[i]).all() == False:
            for j in range(len(retData[i])):
                if np.isfinite(retData[i][j]) == False:
                    if np.isnan(retData[i][j]) == False:
                        print(retData[i][j])
    '''
    retData = np.nan_to_num(retData)
    retData = retData.tolist()
    return retData, retLabels


def incrementCount(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 1

def RunOneFold(dataTrain, dataTest, labelsTrain, labelsTest, queue):
    print(current_process().name + " alive!")
    clf = svm.SVC(kernel='linear').fit(dataTrain, labelsTrain)
    predicted = clf.predict(dataTest)
    #clfModel = svmutil.svm_train(labelsTrain, dataTrain, '-s 0 -t 2 -q -c 1')
    #predicted = svmutil.svm_predict(labelsTest, dataTest, clfModel, '-q')
    #predicted = np.array(predicted[0])
    predicted = np.array(predicted)
    score = np.sum(predicted == labelsTest) / len(labelsTest)
    print("printing one process dist: " + current_process().name)
    print(current_process().name + "----predicted as original: " + str(np.sum(np.array(predicted) == 1)) + " predicted as translated:" + str(np.sum(np.array(predicted) == 0)))
    print(current_process().name + "----labeled as original: " + str(np.sum(np.array(labelsTest) == 1)) + " labeled as translated:" + str(np.sum(np.array(labelsTest) == 0)))
    print(current_process().name + " process score: " + str(score))
    queue.put_nowait(score)

if __name__ == '__main__':
    print( "***Starting Time*** "  + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    parser = argparse.ArgumentParser(description='This script runs an ML trained model according to specified features')
    parser.add_argument('features', help='Search for words or post in corpus POSBigrams POSTrigrams FunctionWords Pronouns',
                        choices=['POSBigrams', 'POSTrigrams', 'FunctionWords', 'Pronouns'])
    parser.add_argument('--pronouns_path', help='Specify the pronouns file path', required=True)
    parser.add_argument('--funcwords_path', help='Specify the function words file path', required=True)
    parser.add_argument('--english_path', help='Specify the english sentences file(tagged,translated&non-translated)', required=True)
    parser.add_argument('--labels_path', help='Specify the english labels file(translated&non-translated)', required=True)
    args = parser.parse_args()
    print(args)
    #english_file_path = "/home/ohadmosafi/NLP_Lab/en_m_tagged.txt"
    #label_file_path = "/home/ohadmosafi/NLP_Lab/fr-en_m.txt"

    if args.features == "Pronouns" or args.features == "FunctionWords":
        if args.features == "Pronouns":
            file_words = readWordsFromFile(args.pronouns_path)
        elif args.features == "FunctionWords":
            file_words = readWordsFromFile(args.funcwords_path)
        valid_words_in_corpus = wordsInCorpus(args.english_path, file_words)
        o, t = divide_to_chnuks(args.english_path, args.labels_path, 'en', valid_words_in_corpus, SearchWordsOrPOS.FIND_WORDS)

    if "POS" in args.features:
        if args.features == "POSTrigrams":
            valid_words_in_corpus = POSinCorpus(args.english_path, 3)
        elif args.features == "POSBigrams":
            valid_words_in_corpus = POSinCorpus(args.english_path, 2)
        o, t = divide_to_chnuks(args.english_path, args.labels_path, 'en', valid_words_in_corpus, SearchWordsOrPOS.FIND_POS)

    data, label = combineSamplesNormalize(o,t)

    print("***Starting MultiProcesses*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("number of samples:" + str(len(label)))
    print("number of original lang samples: " + str(np.sum(np.array(label) == 1)))
    print("number of translated lang samples: " + str(np.sum(np.array(label) == 0)))
    print("number of features:" + str(len(data[0])) + " or " + str(len(valid_words_in_corpus)) )

    folds = 10
    kf = KFold(n_splits=folds, shuffle=True)
    q = Queue()
    processes = []
    for train_index, test_index in kf.split(data):
        X_train = [data[i] for i in train_index]
        X_test = [data[i] for i in test_index]
        y_train = [label[i] for i in train_index]
        y_test = np.array([label[i] for i in test_index])
        p = Process(target=RunOneFold, args=(X_train, X_test, y_train, y_test, q,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    countFolds = 0
    while (not q.empty()):
        countFolds += q.get()
    print ("total success rate is:" + str(countFolds/folds))
    print("***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
