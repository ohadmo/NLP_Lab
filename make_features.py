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
from multiprocessing import Process, Queue
sys.path.append('D:\ohadm\Downloads\libsvm-3.22\python')
import svmutil

CHUNK_SIZE = 2000

def readWordsFromFile(path):
    wordsSet = set()
    with open(path,'r',encoding='utf-8') as file:
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
    valid_POS = set()
    with open(path,'r') as file:
        for line in file:
            pos = [x.strip().split('_')[1] for x in line.split(' ')]
            if len(pos) < ngrams:
                continue
            for i in range(len(pos) - ngrams + 1):
                print(pos[i] + "_" + pos[i+1] + "_" + pos[i+2])

def ChunkToWordsCounters(chunk, cNouns):
    tokens = [x.strip().lower().split('_')[0] for x in chunk]
    pronouns_count = {x: 0 for x in cNouns}
    for token in tokens:
        if token in cNouns:
            pronouns_count[token] += 1
    return [x / len(chunk) for x in pronouns_count.values()]

def divide_to_chnuks(language_file, label_file, lang, corpusNouns):
    original_chunks = []
    original_chunk = []
    translated_chunks = []
    translated_chunk = []
    with open(language_file,'r') as dfile, open(label_file,'r') as lfile:
        for line, label in zip(dfile, lfile):
            tokens = line.split(' ')
            tokens = [x.strip() for x in tokens]
            if label.strip() == lang:
                original_chunk += tokens
                if len(original_chunk) > CHUNK_SIZE:
                    original_chunks.append(ChunkToWordsCounters(original_chunk, corpusNouns))
                    original_chunk = []
            else:
                translated_chunk += tokens
                if len(translated_chunk) > CHUNK_SIZE:
                    translated_chunks.append(ChunkToWordsCounters(translated_chunk, corpusNouns))
                    translated_chunk = []
    original_chunks.append(ChunkToWordsCounters(original_chunk, corpusNouns))
    translated_chunks.append(ChunkToWordsCounters(translated_chunk, corpusNouns))
    return original_chunks, translated_chunks


def incrementCount(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 1

def RunOneFold(dataTrain, dataTest, labelsTrain, labelsTest, queue):
    print("alive!")
    # clf = svm.SVC(kernel='rbf').fit(X_train, y_train)
    # predicted = clf.predict(X_test)
    clfModel = svmutil.svm_train(labelsTrain, dataTrain, '-s 0 -t 2 -q -c 1')
    predicted = svmutil.svm_predict(labelsTest, dataTest, clfModel, '-q')
    predicted = np.array(predicted[0])
    score = np.sum(predicted == y_test) / len(y_test)
    print(score)
    queue.put_nowait(score)

if __name__ == '__main__':
    print( "***Starting Time*** "  + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    pronouns = readWordsFromFile('D:\\GitHub\\NLP_Lab\\pronouns.txt')
    function_words = readWordsFromFile('D:\\GitHub\\NLP_Lab\\function_words.txt')

    english_file_path = "D:\\GitHub\\NLP_Lab\\local_results_afterFix_link_dict\\Results2017-05-23_04-09-16\\openNLP\\en_tagged.txt"
    label_file_path = "D:\\GitHub\\NLP_Lab\\local_results_afterFix_link_dict\\Results2017-05-23_04-09-16\\fr-en.txt"
    #english_file_path = "D:\\GitHub\\NLP_Lab\\en_m_tagged.txt"
    #label_file_path = "D:\\GitHub\\NLP_Lab\\fr-en_m.txt"

    valid_words_in_corpus = wordsInCorpus(english_file_path, function_words)
    #valid_words_in_corpus = POSinCorpus(english_file_path,3)
    o, t = divide_to_chnuks(english_file_path, label_file_path, 'en', valid_words_in_corpus)

    data = []
    label = []
    for chunk in o:
        data.append(chunk)
        label.append(1)
    for chunk in t:
        data.append(chunk)
        label.append(0)
    print("***Starting MultiProcesses*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
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
        countFolds +=  q.get()
    print ("total success rate is:" + str(countFolds/folds))
    print("***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


    #clf = svm.SVC(kernel='linear') # .fit(data, label)
    #print(clf.score(data, label))
    #scores = cross_val_score(clf, data, label, cv=2)
    #print(scores)

    # pronounsCount = {}
    # incrementCount(pronounsCount, token)
