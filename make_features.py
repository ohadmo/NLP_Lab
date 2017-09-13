# go over the files, for every language file, group all original lines together. divide to chunks
# find all pronouns in the corpus
# count pronouns frequencies.
# data points = chunks, dimension = number of pronouns in corpus. values = frequencies. label = {T, O}

import numpy as np
import sys
import datetime
import pickle
import os.path
import operator
from enum import Enum
import argparse


class SearchWordsOrPOS(Enum):
    FIND_WORDS = 0
    FIND_POS = 1

class POSNgram(Enum):
    BIGRAM = 2
    TRIGRAM = 3

def retCurrentTime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def readWordsFromFile(path):
    print(retCurrentTime() + ' In readWordsFromFile: ' + path)
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
    print(retCurrentTime() + ' WordsInCorpus: ' + path)
    valid_words = set()
    with open(path, 'r', encoding='cp1252') as eng_file:
        for line in eng_file:
            if len(valid_words) == len(words_set):
                return valid_words
            tokens = [x.strip().lower().split('_')[0] for x in line.split(' ')]
            for token in tokens:
                if token in words_set:
                    valid_words.add(token)
    print(str(len(valid_words)) + ' valid words from wordsInCorpus, out of ' + str(len(words_set)))
    return valid_words

def POSinCorpus(path, ngrams):
    all_POS = dict()
    with open(path,'r') as file:
        for line in file:
            try:
                # Aftr runing he opennlp toknizer there are some words which are separated by '?' or contains more tha one '_'
                pos_line = [x.strip().rsplit('_',1) if len(x.strip().rsplit('_',1)) == 2 else x.strip().split('?') for x in line.split(' ')]
                pos_line = [x[1] for x in pos_line]
            except:
                print("Exception POSinCorpus: " , line)
            else:
                if len(pos_line) < ngrams:
                    continue
                for i in range(len(pos_line) - ngrams + 1):
                    pos = "_".join([pos_line[i+k] for k in range(ngrams)])
                    if pos not in all_POS:
                        all_POS[pos] = 1
                    else:
                        all_POS[pos] += 1
    return all_POS

def ChunkToWordsCounters(chunk, cNouns, size_of_chunk):
    pronouns_count = {x: 0 for x in cNouns}
    for token in chunk:
        if token in cNouns:
            pronouns_count[token] += 1
    return [((x / len(chunk))*(size_of_chunk/len(chunk))) for x in  pronouns_count.values()]


def divide_to_chnuks(language_file, label_file, lang, corpusNouns, search_enum, pos_n_gram, chunk_size):
    print(retCurrentTime() + "starting divide_to_chnuks")
    original_chunks = []
    original_chunk = []
    translated_chunks = []
    translated_chunk = []
    with open(language_file, 'r', encoding='cp1252') as dfile, open(label_file, 'r', encoding='utf-8') as lfile:
        for line, label in zip(dfile, lfile):
            try:
                # tokens = [x.strip().lower().split('_')[search_enum.value] for x in line.split(' ')]
                tokens = [x.strip().lower().rsplit('_',1) if len(x.strip().rsplit('_',1)) == 2 else x.strip().split('?') for x in line.split(' ')]
                tokens = [x[search_enum.value] for x in tokens]
            except:
                print("exception divide_to_chunks in line:" , line)
            else:
                if search_enum == SearchWordsOrPOS.FIND_POS:
                    t = []
                    for i in range(len(tokens) - pos_n_gram.value + 1):
                        pos = "_".join([tokens[i + k] for k in range(pos_n_gram.value)])
                        t.append(pos.upper())
                    tokens = t
                if label.strip() == lang:
                    original_chunk += tokens
                    if len(original_chunk) > chunk_size:
                        original_chunks.append(ChunkToWordsCounters(original_chunk, corpusNouns, chunk_size))
                        original_chunk = []
                else:
                    translated_chunk += tokens
                    if len(translated_chunk) > chunk_size:
                        translated_chunks.append(ChunkToWordsCounters(translated_chunk, corpusNouns, chunk_size))
                        translated_chunk = []
    original_chunks.append(ChunkToWordsCounters(original_chunk, corpusNouns, chunk_size))
    translated_chunks.append(ChunkToWordsCounters(translated_chunk, corpusNouns, chunk_size))
    return original_chunks, translated_chunks


def incrementCount(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 1

def PosFindTokensAllLanguages(languages, ngram, work_path, tokenized_data_file_name, n_most_freq):
    valid_pos_in_corpus = dict()
    for lang in languages:
        pos_pickle_path = os.path.join(work_path, lang, str(ngram)+"pos.p")
        if os.path.isfile(pos_pickle_path):
            print(retCurrentTime() + "***loading from pickle in path" + str(pos_pickle_path))
            with open(pos_pickle_path, 'rb') as f:
                valid_words_in_one = pickle.load(f)
        else:
            print(retCurrentTime() + "***In lang " + lang + " calling POSinCorpus and saving POS words to pickle")
            valid_words_in_one = POSinCorpus(os.path.join(work_path, lang, tokenized_data_file_name), ngram)
            with open(pos_pickle_path, 'wb') as f:
                pickle.dump(valid_words_in_one, f)
        for k, v in valid_words_in_one.items():
            if k in valid_pos_in_corpus:
                valid_pos_in_corpus[k] += v
            else:
                valid_pos_in_corpus[k] = v
    ret_valid_POS = sorted(valid_pos_in_corpus.items(), key=operator.itemgetter(1), reverse=True)  # Decreasing Order
    temp = [x[0] for x in ret_valid_POS][:n_most_freq]
    return set(temp)

def FuncPronounsFindTokensAllLanguages(words_file_path, languages, pickle_name, work_path, tokenized_data_file_name):
    ret_valid_words_in_corpus = set()
    file_words = readWordsFromFile(words_file_path)
    for lang in languages:
        pickle_path = os.path.join(work_path, lang, pickle_name)
        if(os.path.isfile(pickle_path)):
            print(retCurrentTime() + "***loading from pickle in path " + str(pickle_path))
            with open(pickle_path, 'rb') as f:
                ret_valid_words_in_corpus = ret_valid_words_in_corpus | pickle.load(f)
        else:
            print(retCurrentTime() + "***calling wordsInCorpus and saving to pickle " + str(pickle_path))
            words_in_one_corpus = wordsInCorpus(os.path.join(work_path, lang, tokenized_data_file_name), file_words)
            with open(pickle_path,'wb') as f:
                pickle.dump(words_in_one_corpus, f)
            ret_valid_words_in_corpus = ret_valid_words_in_corpus | words_in_one_corpus
    return ret_valid_words_in_corpus


def CreateFeatureVectors(work_path, language, output_original_data_name, output_translated_data_name,
                         lang_data_tagged_name, labeled_data_name, validTokensSet, wordsOrPos, pos_n_gram, size_of_chunk):
    dataOrg_file = os.path.join(work_path, language, output_original_data_name)
    dataTrans_file = os.path.join(work_path, language, output_translated_data_name)
    if os.path.isfile(dataOrg_file) and os.path.isfile(dataTrans_file):
        print(retCurrentTime() + "***loading from pickle in path: " + str(dataOrg_file) + " " + str(dataTrans_file))
        with open(dataOrg_file, 'rb') as df:
            data = pickle.load(df)
        with open(dataTrans_file, 'rb') as lf:
            label = pickle.load(lf)
    else:
        print(retCurrentTime() + "***calling divide to chunks and dumping: " + str(dataOrg_file) + " " + str(dataTrans_file))
        data, label = divide_to_chnuks(os.path.join(work_path, language, lang_data_tagged_name),
                                os.path.join(work_path, language, labeled_data_name),
                                'en', validTokensSet, wordsOrPos, pos_n_gram, size_of_chunk)
        print("no. of sampels in original: " + str(len(data)))
        print("no. of sampels in translated: " + str(len(label)))
        pickle.dump(np.array(data), open(dataOrg_file, 'wb'))
        pickle.dump(np.array(label), open(dataTrans_file, 'wb'))
    return data, label


def createCombinedLanguagesOrgTransCorpora(org_out_path, trans_out_path, working_dir, languages,
                                           tagged_file_name_each_language,
                                           labeled_file_halfname_each_language, original_lang):
    print(retCurrentTime() + " Starting createCombinedLanguagesOrgTransCorpora")
    org_combined = open(org_out_path, 'w+', encoding='cp1252')
    trans_combined = open(trans_out_path, 'w+', encoding='cp1252')

    data_files = []
    labels_files = []

    done_list_data_files = []
    done_list_labels_files = []

    for lang in languages:
        df = open(os.path.join(working_dir, lang, tagged_file_name_each_language), 'r', encoding='cp1252')
        lf = open(os.path.join(working_dir, lang, lang + labeled_file_halfname_each_language), 'r', encoding='utf-8')
        data_files.append(df)
        labels_files.append(lf)

    while (data_files != []):
        n = np.random.randint(0, len(data_files))
        write_data = data_files[n].readline()
        write_label = labels_files[n].readline()

        if write_data == '' or write_label == '':
            done_list_data_files.append(data_files.pop(n))
            done_list_data_files.append(labels_files.pop(n))
        else:
            if write_label.strip() == original_lang:
                org_combined.write(write_data)
            else:
                trans_combined.write(write_data)
    org_combined.close()
    trans_combined.close()
    for cf, cl in zip(done_list_data_files, done_list_labels_files):
        cf.close()
        cl.close()
    print(retCurrentTime() + " Finished createCombinedLanguagesOrgTransCorpora")


def getLineFeatures(line, search_enum, ngrams):
    try:
        # tokens = [x.strip().lower().split('_')[search_enum.value] for x in line.split()]
        tokens = [x.strip().lower().rsplit('_', 1) if len(x.strip().rsplit('_', 1)) == 2 else x.strip().split('?') for x in line.split(' ')]
        tokens = [x[search_enum.value] for x in tokens]
    except:
        print("exception getLineFeatures:", line)
    else:
        if search_enum == SearchWordsOrPOS.FIND_POS:
            line_trigram = []
            for i in range(len(tokens) - ngrams + 1):
                pos_trigram = "_".join([tokens[i + k] for k in range(ngrams)])
                line_trigram.append(pos_trigram.upper())
            tokens = line_trigram
        return tokens


def getDataPoints(file_path, corpusNouns, search_enum, TriOrBi, size_of_chunk):
    with open(file_path, 'r', encoding='cp1252') as file:
        chunks = list()
        current_chunk = list()
        for line in file:
            line_trigram = getLineFeatures(line, search_enum, TriOrBi)
            if len(current_chunk) <= size_of_chunk:
                current_chunk.extend(line_trigram)
            else:
                chunks.append(ChunkToWordsCounters(current_chunk, corpusNouns, size_of_chunk))
                current_chunk = list()
        return chunks

def createSamplesPickleFromCombnidCorpora(org_combined_path, trans_combined_path, corpusNouns, SearchWordOrPos,
                                          ngrams, size_of_chunk, working_dir, output_org_pickle_name,
                                          output_trans_pickle_name):
    print("Start createSamplesPickleFromCombnidCorpora" + retCurrentTime())
    original_data = getDataPoints(org_combined_path, corpusNouns, SearchWordOrPos, ngrams, size_of_chunk)
    translated_data = getDataPoints(trans_combined_path, corpusNouns, SearchWordOrPos, ngrams, size_of_chunk)
    print("no. of sampels in original: " + str(len(original_data)))
    print("no. of sampels in translated: " + str(len(translated_data)))
    pickle.dump(original_data, open(os.path.join(working_dir, output_org_pickle_name), 'wb'))
    pickle.dump(translated_data, open(os.path.join(working_dir, output_trans_pickle_name), 'wb'))
    print("finished createSamplesPickleFromCombnidCorpora" + retCurrentTime())

if __name__ == '__main__':
    print( "***Starting Time*** "  + retCurrentTime())
    parser = argparse.ArgumentParser(description='This script extract features from the data and av to pickle')
    parser.add_argument('features', help='Search for words or post in corpus POSBigrams POSTrigrams FunctionWords Pronouns',
                        choices=['POSBigrams', 'POSTrigrams', 'FunctionWords', 'Pronouns'])
    parser.add_argument('mode', help='Two options are available: '
                                     'CombinedLanguageChunks(CLC) Original and Translated chunks are built from lines '
                                     'from different languages. '
                                     'SeparatedLanguageChunks(SLC) chunks are built for each language separately.',
                        choices=['CLC', 'SLC'])
    parser.add_argument('--pronouns_path', help='Specify the pronouns file path', required=True)
    parser.add_argument('--funcwords_path', help='Specify the function words file path', required=True)
    parser.add_argument('--dicts_path', help="dicts folder path", required=True)
    parser.add_argument('--languages', help="languages to apply this scrip on separated by a blank space", required=True)
    parser.add_argument('--pos_top_n', help="specify the number of the top frequent  Bigrams/Trigrams to pick in all corpora ", required=True)
    parser.add_argument('--chunk_size',help="defines the chunk size", required=True)
    args = parser.parse_args()

    if args.features == "Pronouns" or args.features == "FunctionWords":
        if args.features == "Pronouns":
            valid_words_in_corpus = FuncPronounsFindTokensAllLanguages(args.pronouns_path, args.languages.split(" "),
                                                                       'pronouns_words.p', args.dicts_path,'en_tagged_tweetTokenizer.txt')
        else: # args.features equals "FunctionWords"
            valid_words_in_corpus = FuncPronounsFindTokensAllLanguages(args.funcwords_path, args.languages.split(" "),
                                                                       'function_words.p', args.dicts_path, 'en_tagged_tweetTokenizer.txt')
        if args.mode == "SLC":
            for lang in args.languages.split(" "):
                data, label = CreateFeatureVectors(args.dicts_path, lang, 'dataOriginal_{0}.p'.format(str.lower(args.features)),
                                                   'dataTranslated_{0}.p'.format(str.lower(args.features)),
                                                   'en_tagged_tweetTokenizer.txt', lang + "-en.txt", valid_words_in_corpus,
                                                   SearchWordsOrPOS.FIND_WORDS, None, int(args.chunk_size))
        else: # args.mode equals "CLC"
            org_combined_path = 'C:\\NLP\\combinedOrg.txt'
            trans_combined_path = 'C:\\NLP\\combinedTrans.txt'
            if not os.path.isfile(org_combined_path) and not os.path.isfile(trans_combined_path):
                createCombinedLanguagesOrgTransCorpora(org_combined_path, trans_combined_path,
                                                       args.dicts_path, args.languages.split(" "),
                                                       'en_tagged_tweetTokenizer.txt', "-en.txt", 'en')
            else:
                print("CombinedCombinedLanguagesOrgTransCorpora already exists")
            createSamplesPickleFromCombnidCorpora(org_combined_path, trans_combined_path, valid_words_in_corpus,
                                                  SearchWordsOrPOS.FIND_WORDS,
                                                  None, int(args.chunk_size), args.dicts_path,
                                                  'all_original_samples_{0}.p'.format(str.lower(args.features)),
                                                  'all_translated_samples_{0}.p'.format(str.lower(args.features)))
    if "POS" in args.features:
        chosenNgram = None
        if args.features == "POSTrigrams":
            chosenNgram = POSNgram.TRIGRAM
        else:
            chosenNgram = POSNgram.BIGRAM

        valid_POS = PosFindTokensAllLanguages(args.languages.split(' '), chosenNgram.value,
                                              args.dicts_path, 'en_tagged_tweetTokenizer.txt', int(args.pos_top_n))
        if args.mode == "SLC":
            for lang in args.languages.split(" "):
                data, label = CreateFeatureVectors(args.dicts_path, lang, 'dataOriginal_{0}pos.p'.format(chosenNgram.value),
                                                   'dataTranslated_{0}pos.p'.format(chosenNgram.value),
                                                   'en_tagged_tweetTokenizer.txt', lang+"-en.txt", valid_POS,
                                                   SearchWordsOrPOS.FIND_POS, chosenNgram, int(args.chunk_size))
        else: #args.mode equals 'CLC'
            org_combined_path = 'C:\\NLP\\combinedOrg.txt'
            trans_combined_path = 'C:\\NLP\\combinedTrans.txt'
            if not os.path.isfile(org_combined_path) and not os.path.isfile(trans_combined_path):
                createCombinedLanguagesOrgTransCorpora(org_combined_path, trans_combined_path,
                                                       args.dicts_path, args.languages.split(" "),
                                                       'en_tagged_tweetTokenizer.txt', "-en.txt", 'en')
            else:
                print("CombinedCombinedLanguagesOrgTransCorpora already exists")
            createSamplesPickleFromCombnidCorpora(org_combined_path, trans_combined_path, valid_POS,
                                                  SearchWordsOrPOS.FIND_POS,
                                                  chosenNgram.value, int(args.chunk_size), args.dicts_path,
                                                  'all_original_samples_{0}grams.p'.format(chosenNgram.value),
                                                  'all_translated_samples_{0}grams.p'.format(chosenNgram.value))
        # else: # args.features equals "POSBigrams"
        #     valid_POS = PosFindTokensAllLanguages(args.languages.split(' '), 2,
        #                                           args.dicts_path, 'en_tagged_tweetTokenizer.txt', int(args.pos_top_n))
        #     if args.mode == "SLC":
        #         for lang in args.languages.split(" "):
        #             data, label = CreateFeatureVectors(args.dicts_path, lang, 'dataOriginal_2pos.p', 'dataTranslated_2pos.p',
        #                                                'en_tagged_tweetTokenizer.txt', lang+"-en.txt", valid_POS,
        #                                                SearchWordsOrPOS.FIND_POS, POSNgram.BIGRAM, int(args.chunk_size))
        #     else: #args.mode equals 'CLC'
        #         org_combined_path = 'C:\\NLP\\combinedOrg.txt'
        #         trans_combined_path = 'C:\\NLP\\combinedTrans.txt'
        #         if not os.path.isfile(org_combined_path) and not os.path.isfile(trans_combined_path):
        #             createCombinedLanguagesOrgTransCorpora(org_combined_path, trans_combined_path,
        #                                                    args.dicts_path, args.languages.split(" "),
        #                                                    'en_tagged_tweetTokenizer.txt', "-en.txt", 'en')
        #         else:
        #             print("CombinedCombinedLanguagesOrgTransCorpora already exists")
        #         createSamplesPickleFromCombnidCorpora(org_combined_path, trans_combined_path, valid_POS,
        #                                               SearchWordsOrPOS.FIND_POS,
        #                                               2, args.chunk_size, args.dicts_path,
        #                                               'all_original_samples_bigrams.p',
        #                                               'all_translated_samples_bigrams.p')
    print("***Finish Time*** " + retCurrentTime())