import argparse
import pickle
import numpy as np
from sklearn import svm, linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from multiprocessing import Process, Queue, current_process
import datetime
import os
import sys
#sys.path.append('D:\ohadm\Downloads\libsvm-3.22\python')
#import svmutil
from make_features import retCurrentTime


def NormalizeData(data):
    print("Normalizing the entire combined data")
    data[:] = preprocessing.scale(data)
    data[:] = preprocessing.normalize(data)

def getDesiredClassifier(classifier_name):
    def getSVM():
        print("SVM classifier is chosen!")
        return svm.SVC(kernel='linear')
    def getLogisticRegression():
        print("Logistic Regression classifier is chosen!")
        return linear_model.LogisticRegression(C=2)
    def getRandomForest():
        print("Random Forest classifier is chosen!")
        return RandomForestClassifier(max_depth=200, random_state=0)
    return {
        'linear_svm' :          getSVM,
        'logistic_regression':  getLogisticRegression,
        'random_forest':        getRandomForest
    }[classifier_name]()

def RunOneFold(dataTrain, dataTest, labelsTrain, labelsTest, queue, classifier_obj):
    print(current_process().name + " alive!")
    print("starting train " + current_process().name)
    clf = classifier_obj.fit(dataTrain, labelsTrain)
    print("training done " + current_process().name)
    predicted = clf.predict(dataTest)
    predicted = np.array(predicted)
    # dataTrain = [list(x) for x in dataTrain]
    # dataTest = [list(x) for x in dataTest]
    # clfModel = svmutil.svm_train(labelsTrain, dataTrain, '-s 1 -t 0 -q ')
    # predicted = svmutil.svm_predict(labelsTest, dataTest, clfModel, '-q')
    # predicted = np.array(predicted[0])
    score = np.sum(predicted == labelsTest) / len(labelsTest)
    confMat = np.zeros((2,2),dtype=np.int)
    for i in range(len(labelsTest)):
        confMat[int(labelsTest[i]), int(predicted[i])] += 1
    print("*****{0} - Conffusion Matrix*****".format(current_process().name))
    c = np.concatenate(( [["Trns", "Org"]], confMat), axis=0)
    c = np.concatenate(( np.array([["^^^^", "Trns", "Org"]]).T, c),axis=1)
    c = np.concatenate(( [["^^^^", "Pred", "^^^^"]], c), axis=0)
    c = np.concatenate(( np.array([["^^^^", "^^^^", "Actu","^^^^"]]).T, c),axis=1)
    print(c)
    print("******************")
    print(current_process().name + "printing  process distribution")
    print(current_process().name + "----predicted as original: " + str(np.sum(np.array(predicted) == 1)) + " predicted as translated:" + str(np.sum(np.array(predicted) == 0)))
    print(current_process().name + "----labeled as original: " + str(np.sum(np.array(labelsTest) == 1)) + " labeled as translated:" + str(np.sum(np.array(labelsTest) == 0)))
    print(current_process().name + " process score: " + str(score))
    queue.put_nowait(score)

def returnShuffledIdx(numOfObjs):
    shuffledIdx = np.arange(numOfObjs)
    np.random.shuffle(shuffledIdx)
    return shuffledIdx

def CreateBalancedData(fileName, num_samples_per_lang, num_features, langs_list, work_dir, zero_one_label):
    data = np.empty((num_samples_per_lang * len(langs_list), num_features))
    labels = np.empty(num_samples_per_lang * len(langs_list))
    for i in range(len(langs_list)):
        try:
            with open(os.path.join(work_dir, langs_list[i], fileName),'rb') as lang_file:
                one_lang_data = pickle.load(lang_file)
                print("!!! In path " + lang_file.name + ", choosing randomly " + str(num_samples_per_lang) + " out of " + str(len(one_lang_data)) + " samples")
                randomIncs = returnShuffledIdx(one_lang_data.shape[0])
                randomIncs = randomIncs[0:num_samples_per_lang]
                data[i * num_samples_per_lang : (i + 1) * num_samples_per_lang] = one_lang_data[randomIncs, :]
                labels[i * num_samples_per_lang: (i + 1) * num_samples_per_lang] = np.full(num_samples_per_lang,zero_one_label)
        except FileNotFoundError:
            print("Error: In bilingual corpus dir {0} there is no artifact from the feature extraction \n"
                  "script in expected location {1}".format(lang,os.path.join(work_dir, langs_list[i], fileName)))
            sys.exit(1)
    return data, labels


def get_data_and_labels(translated_name_file_path, original_name_file_path,
                        sample_per_lang, features_num, langs_list, work_dir):
    translated_data, translated_labels = \
        CreateBalancedData(translated_name_file_path, sample_per_lang, features_num, langs_list, work_dir, 0)
    original_data, original_labels = \
        CreateBalancedData(original_name_file_path, sample_per_lang, features_num, langs_list, work_dir,  1)
    data = np.concatenate([original_data, translated_data])
    label = np.concatenate([original_labels, translated_labels])
    return data, label


def get_data_and_label_from_p(file_name, dir, label):
    with open(os.path.join(dir, file_name), 'rb') as file:
        samples = np.array(pickle.load(file))
        labels =  np.full(len(samples), label)
        return samples, labels

def createParser():
    parser = argparse.ArgumentParser(description='This script is responsible for running a classifier using 10fold'
                                                 ' cross validations. ')
    parser.add_argument('mode', help='Two options are available:(according to the running mode of make_features.py) '
                                     'CombinedLanguageChunks(CLC) Original and Translated chunks are built from lines '
                                     'of different languages. '
                                     'SeparatedLanguageChunks(SLC) chunks are built for each language separately.',
                        choices=['CLC', 'SLC'])
    parser.add_argument('classifier', help="Desired classifier to be chosen",
                        choices=['linear_svm', 'logistic_regression', 'random_forest'])
    parser.add_argument('--working_dir', help='Specify the working dir path', required=True)
    parser.add_argument('--translated_samples_file_name',
                        help='In CombinedLanguageChunks mode: specify The name of the translated samples file '
                             'which is located in each of the language dirs under working dir path '
                             'In SeparatedLanguageChunks mode: specify the name of the combined translated file '
                             'which should be located under working dir', required=True)
    parser.add_argument('--original_samples_file_name',
                        help='In CombinedLanguageChunks mode: specify The name of the original samples file '
                             'which is located in each of the language dirs under working dir path '
                             'In SeparatedLanguageChunks mode: specify the name of the combined original file '
                             'which should be located under working dir', required=True)
    parser.add_argument('--languages', help="languages to apply this script on separated by a blank space",
                        required=False)
    return parser

if __name__ == '__main__':
    args = createParser().parse_args()
    if args.mode == "SLC":
        print(retCurrentTime() + " Start Loading data Separated languages Corpora")
        # Loading translated data samples from each language folder, the num of samples taken from each folder is determined
        # by the language with the least amount of translated samples in it
        sample_per_lang = float('inf')
        features_num = None
        for lang in args.languages.split(" "):
            try:
                with open(os.path.join(args.working_dir, lang, args.translated_samples_file_name), 'rb') as transFile:
                    one_lang_translated_data = np.array(pickle.load(transFile))
                    sample_per_lang = one_lang_translated_data.shape[0] if (one_lang_translated_data.shape[0] < sample_per_lang) else sample_per_lang
                    features_num = one_lang_translated_data.shape[1]
            except FileNotFoundError:
                print("Error: In bilingual corpus dir {0} there is no artifact from the feature extraction \n"
                      "script in expected location {1}".format(lang,os.path.join(args.working_dir, lang, args.translated_samples_file_name)))
                sys.exit(1)
        # Creating balanced dataset
        data, label = get_data_and_labels(args.original_samples_file_name, args.translated_samples_file_name,
                                          sample_per_lang, features_num, args.languages.split(" "), args.working_dir)
    else: #args.mode equals "CLC"
        print(retCurrentTime() + " Start Loading data Combined languages Corpora")
        translated_data, translated_labels = \
            get_data_and_label_from_p(args.translated_samples_file_name, args.working_dir, 0)
        original_data, original_labels = \
            get_data_and_label_from_p(args.original_samples_file_name, args.working_dir, 1)
        print("after loading combined dataset: translated samples={0} and original samples={1}".format(len(translated_labels),len(original_labels))  )

        sidx = returnShuffledIdx(len(translated_labels))
        original_data=original_data[sidx,:]
        original_labels = original_labels[sidx]
        print("in combined original samples were picked randomly and now original samples={0}".format(len(original_labels)))

        data = np.concatenate((translated_data, original_data))
        label = np.concatenate((translated_labels, original_labels))

    # Shuffling after concatenating original & translated samples
    shuffled_idx = returnShuffledIdx(len(label))
    data = data[shuffled_idx,:]
    label = label[shuffled_idx]

    # Standardizing & Normalizing Data
    NormalizeData(data)

    print("*** About to start Multiclass classification *** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("number of samples:" + str(label.shape))
    print("number of original lang samples: " + str(np.sum(np.array(label) == 1)))
    print("number of translated lang samples: " + str(np.sum(np.array(label) == 0)))
    print("number of features:" + str(len(data[0])))
    folds = 10
    kf = KFold(n_splits=folds, shuffle=True)
    q = Queue()
    processes = []
    chosen_classifier = getDesiredClassifier(args.classifier)
    for train_index, test_index in kf.split(data):
        X_train = [data[i] for i in train_index]
        X_test = [data[i] for i in test_index]
        y_train = [label[i] for i in train_index]
        y_test = np.array([label[i] for i in test_index])
        p = Process(target=RunOneFold, args=(X_train, X_test, y_train, y_test, q, chosen_classifier))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    calcL = []
    while (not q.empty()):
        tempVal = q.get()  * 100
        calcL.append(tempVal)
    print("***Average success rate is:************************" + str(np.mean(calcL)))
    print("***STD is :*** " + str(np.std(calcL)))
    print(calcL)
    print("***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


