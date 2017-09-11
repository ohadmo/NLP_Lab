import pickle
import numpy as np
from sklearn import svm, linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from multiprocessing import Process, Queue, current_process
import datetime
import sys
import os
sys.path.append('D:\ohadm\Downloads\libsvm-3.22\python')
import svmutil


sample_per_lang = float('inf')
features_num = None
workDir = 'C:\\NLP\\'
translated_name_file_path = 'dataTranslated_pos.p'
original_name_file_path = 'dataOriginal_pos.p'
langs = ['fr','es','ar','ru']
#langs =['ar','ru']


def NormalizeData(data):
    print("Normalizing the entire combined data")
    # mean = np.sum(data, axis=0) / len(data)
    # temp = -1* np.tile(mean,(len(data),1))
    # variance = (np.array(data) + temp)
    # variance = variance**2
    # variance = np.sum(variance, axis=0) / len(data)
    # variance = np.sqrt(variance)
    # data[:] = (data-mean)/variance
    #
    # _min = np.min(data, axis=0)
    # _max = np.max(data, axis=0)
    # data[:] = (data - _min) / (_max - _min)

    # print("Infinite indices:")
    # for i in range(len(retData)):
    #     if np.isfinite(retData[i]).all() == False:
    #         for j in range(len(retData[i])):
    #             if np.isfinite(retData[i][j]) == False:
    #                 if np.isnan(retData[i][j]) == False:
    #                     print(retData[i][j])

    # data[:] = np.nan_to_num(data)
    data[:] = preprocessing.scale(data)
    data[:] = preprocessing.normalize(data)

def RunOneFold(dataTrain, dataTest, labelsTrain, labelsTest, queue):
    print(current_process().name + " alive!")
    print("starting train " + current_process().name)
    clf = svm.SVC(kernel='linear').fit(dataTrain, labelsTrain)
    #clf =  linear_model.LogisticRegression(C=2).fit(dataTrain, labelsTrain)
    #clf =  RandomForestClassifier(max_depth=200, random_state=0).fit(dataTrain, labelsTrain)
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
    print("*****conf mat*****")
    c = np.concatenate(( [["Trns", "Org"]], confMat), axis=0)
    c = np.concatenate(( np.array([["^^^^", "Trns", "Org"]]).T, c),axis=1)
    c = np.concatenate(( [["^^^^", "Pred", "^^^^"]], c), axis=0)
    c = np.concatenate(( np.array([["^^^^", "^^^^", "Actu","^^^^"]]).T, c),axis=1)
    print(c)
    print("******************")
    print("printing one process dist: " + current_process().name)
    print(current_process().name + "----predicted as original: " + str(np.sum(np.array(predicted) == 1)) + " predicted as translated:" + str(np.sum(np.array(predicted) == 0)))
    print(current_process().name + "----labeled as original: " + str(np.sum(np.array(labelsTest) == 1)) + " labeled as translated:" + str(np.sum(np.array(labelsTest) == 0)))
    print(current_process().name + " process score: " + str(score))
    queue.put_nowait(score)

def returnShuffledIdx(numOfObjs):
    shuffledIdx = np.arange(numOfObjs)
    np.random.shuffle(shuffledIdx)
    return shuffledIdx

def CreateBalancedData(fileName, num_samples_per_lang, num_features, zero_one_label):
    data = np.empty((num_samples_per_lang * len(langs), num_features))
    labels = np.empty(num_samples_per_lang * len(langs))
    for i in range(len(langs)):
        with open(os.path.join(workDir,langs[i],fileName),'rb') as lang_file:
            one_lang_data = pickle.load(lang_file)
            print("!!! In path " + lang_file.name + ", choosing randomly " + str(num_samples_per_lang) + " out of " + str(len(one_lang_data)))
            randomIncs = returnShuffledIdx(one_lang_data.shape[0])
            randomIncs = randomIncs[0:num_samples_per_lang]
            data[i * num_samples_per_lang : (i + 1) * num_samples_per_lang] = one_lang_data[randomIncs, :]
            labels[i * num_samples_per_lang: (i + 1) * num_samples_per_lang] = np.full(num_samples_per_lang,zero_one_label)
    return data, labels

if __name__ == '__main__':
    # Loading translated data samples from each language folder, the num of samples taken from each folder is determined
    # by the language with the least amount of translated samples in it
    for i in range(len(langs)):
        with open('C:\\NLP\\' + langs[i] + '\\dataTranslated_pos.p', 'rb') as transFile:
            one_lang_translated_data = np.array(pickle.load(transFile))
            sample_per_lang = one_lang_translated_data.shape[0] if (one_lang_translated_data.shape[0] < sample_per_lang) else sample_per_lang
            features_num = one_lang_translated_data.shape[1]

    # Creating balanced dataset
    translated_data, translated_labels = CreateBalancedData(translated_name_file_path, sample_per_lang, features_num, 0)
    original_data, original_labels = CreateBalancedData(original_name_file_path, sample_per_lang, features_num, 1)
    data = np.concatenate([original_data, translated_data])
    label = np.concatenate([original_labels, translated_labels])

    # Shuffling after concatenating original & translated samples
    shuffled_idx = returnShuffledIdx(len(label))
    data = data[shuffled_idx,:]
    label = label[shuffled_idx]

    #Normalzing Data
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
    calcL = []
    while (not q.empty()):
        tempVal = q.get()  * 100
        countFolds += tempVal
        calcL.append(tempVal)
    print ("***total success rate is:************************" + str(countFolds/folds))
    print("***total success rate2 is:************************" + str(np.mean(calcL)))
    print("***std is :*** " + str(np.std(calcL)))
    print(calcL)
    print("***Finish Time*** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


