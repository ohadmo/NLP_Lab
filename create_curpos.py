# createa new files - english, french, index
# for every file in the english database
# if the file has an original tag
# if the file has a french translation
# for every <s> tag in the file:
# add the content of the <s> tag to the file "english"
# add the appropriate line to the file "french"
# add the appropriate line to the file "index"

import os
import xml.etree.ElementTree as ET
import glob
import sys
from datetime import datetime

# Desired 3 files output folder
result_output_folder = "D:\\GitHub\\NLP_Lab\\Results"

# The root directory of the data. The root directory contains all the language folders and the links folders
root_path = "D:\\GitHub\\NLP_Lab\\"

# The language name abbreviation
languages = ["en", "fr", "es", "ar", "zh", "ru"]

# A dictionary matching every language to the path of the folder containing the documents of the language
lang_to_path = {x: os.path.join(root_path, x) for x in languages}

# The name of the links folders
links = ["es_en", "ar_en", "fr_en", "ru_en", "zh_en"]

# A dictionary of each language pair to the link directory
language_pair_to_path = {x: os.path.join(root_path, "links", x) for x in links}


en_val =\
    [(['english'],'en'),
     (['french'],'fr'),
     (['spanish'],'es'),
     (['russian'], 'ru'),
     (['chinese'], 'zh'),
     (['arabic'], 'ar')]

languages_dict = {
    'fr' : [(['anglais'],'en'),
            (['français', 'francais'],'fr'),
            (['espagnol', 'espâgnol'],'es'),
            (['russe'],'ru'),
            (['chinois'],'zh'),
            (['arabe'], 'ar')],
    'en' : en_val,
    'es': [(['ingles','inglés'], 'en'),
           (['frances','francés'], 'fr'),
           (['español'], 'es'),
           (['ruso'], 'ru'),
           (['chino'], 'zh'),
           (['arabe','árabe'], 'ar')],
    'ru': en_val,    # some original languages are filtered because the russian alphabet is used for the letters: H B A
    'ar': en_val,
    'zh': en_val
}


class CorpusStatistics:
    def __init__(self):
        # Protocols
        self.total_number_of_protocols = 0
        self.number_of_valid_protocols = 0
        self.number_protocols_unobtainable_xml_first_root = 0
        self.number_protocols_unobtainable_xml_second_root = 0
        self.number_protocols_unobtainable_file_not_found = [0]*2
        self.number_protocols_unobtainable_xml_unparseable = [0]*2

        # Origin language Extraction
        self.number_protocols_unobtainable_origin_lang = 0
        self.number_protocols_not_extracted_origin_both = 0
        self.number_protocols_not_extracted_origin_only_first = 0
        self.number_protocols_not_extracted_origin_only_second = 0
        self.number_protocols_extracted_origin_conflicted = 0
        # =========== unimportant shit ===========
        self.number_protocols_not_contain_original_tag = [0]*2
        self.number_protocols_origin_is0 = [0]*2
        self.number_protocols_origin_is1 = [0]*2
        self.number_protocols_origin_is2 = [0]*2
        self.number_protocols_origin_is_more = [0]*2
        # ========================================

        #Origin language matching
        self.number_protocols_origin_lang_not_any_of_two = 0
        self.number_of_protocols_not_obtainable_link_file = 0
        # valid protocols
        self.number_of_protocols_originally_first_lang = 0
        self.number_of_protocols_originally_second_lang = 0

        # Sentences
        self.total_number_of_sentences = 0
        self.number_of_valid_sentences = 0
        self.number_of_sentences_not_match_lang1 = 0
        self.number_sentences_not_found_in_link_file = 0
        self.number_of_sentences_aligned_id_from_links_not_found_second_file = 0
        self.number_of_sentences_not_match_lang2 = 0
        self.number_of_sentences_writing_exception = 0
        self.number_of_sentences_in_originally_first_lang = 0
        self.number_of_sentences_in_originally_second_lang = 0


    def print_stat(self):
        print("=================================PROTOCOLS=================================")
        print("Total number of protocols:{0} out of them {1} valid (in use) protocols, the rate is {2}".format(
            self.total_number_of_protocols, self.number_of_valid_protocols,
            str(self.number_of_valid_protocols / self.total_number_of_protocols)))
        print("{0} protocols were filtered due to unobtainable xml root of the first file. rate: {1}".format(
            self.number_protocols_unobtainable_xml_first_root,
            str(self.number_protocols_unobtainable_xml_first_root / self.total_number_of_protocols)))
        print(">>> {0} protocols were filtered due to first file not found. rate: {1}".format(
            self.number_protocols_unobtainable_file_not_found[0],
            str(self.number_protocols_unobtainable_file_not_found[0] / self.total_number_of_protocols)))
        print(">>> {0} protocols were filtered due to unable to parse the first file as an xml. rate: {1}".format(
            self.number_protocols_unobtainable_xml_unparseable[0],
            str(self.number_protocols_unobtainable_xml_unparseable[0] / self.total_number_of_protocols)))
        print("{0} protocols were filtered due to unobtainable xml root of the second file. rate: {1}".format(
            self.number_protocols_unobtainable_xml_second_root,
            str(self.number_protocols_unobtainable_xml_second_root / self.total_number_of_protocols)))
        print(">>> {0} protocols were filtered due to second file not found. rate: {1}".format(
            self.number_protocols_unobtainable_file_not_found[1],
            str(self.number_protocols_unobtainable_file_not_found[1] / self.total_number_of_protocols)))
        print(">>> {0} protocols were filtered due to unable to parse the second file as an xml. rate: {1}".format(
            self.number_protocols_unobtainable_xml_unparseable[1],
            str(self.number_protocols_unobtainable_xml_unparseable[1] / self.total_number_of_protocols)))

        print("{0} protocols were filtered due to unobtainable origin language. rate: {1}".format(
            self.number_protocols_unobtainable_origin_lang,
            str(self.number_protocols_unobtainable_origin_lang / self.total_number_of_protocols)))
        print(">>> {0} origin language from both files could not be extracted. rate:{1}".format(
            self.number_protocols_not_extracted_origin_both,
            str(self.number_protocols_not_extracted_origin_both / self.total_number_of_protocols)))
        print(">>> {0} origin language only from first file could not be extracted. rate:{1}".format(
            self.number_protocols_not_extracted_origin_only_first,
            str(self.number_protocols_not_extracted_origin_only_first / self.total_number_of_protocols)))
        print(">>> {0} origin language only from second file could not be extracted. rate:{1}".format(
            self.number_protocols_not_extracted_origin_only_second,
            str(self.number_protocols_not_extracted_origin_only_second / self.total_number_of_protocols)))
        print(">>> {0} protocols were filtered due to conflict between the extracted origin. rate: {1}".format(
            self.number_protocols_extracted_origin_conflicted,
            str(self.number_protocols_extracted_origin_conflicted/self.total_number_of_protocols)))


        print("--->{0} protocols not contain original tag from first language folder. rate: {1}".format(
            self.number_protocols_not_contain_original_tag[0],
            str(self.number_protocols_not_contain_original_tag[0] / self.total_number_of_protocols)))
        print("--->{0} protocols not contain original tag from second language folder. rate: {1}".format(
            self.number_protocols_not_contain_original_tag[1],
            str(self.number_protocols_not_contain_original_tag[1] / self.total_number_of_protocols)))
        print("--->{0} protocols zero origin lang extracted from first language folder files. rate: {1}".format(
            self.number_protocols_origin_is0[0],
            str(self.number_protocols_origin_is0[0] / self.total_number_of_protocols)))
        print("--->{0} protocols zero origin lang extracted from second language folder files. rate: {1}".format(
            self.number_protocols_origin_is0[1],
            str(self.number_protocols_origin_is0[1] / self.total_number_of_protocols)))
        print("--->{0} protocols have two origin lang extracted from first language folder files. rate: {1}".format(
            self.number_protocols_origin_is2[0],
            str(self.number_protocols_origin_is2[0] / self.total_number_of_protocols)))
        print("--->{0} protocols have two origin lang extracted from second language folder files. rate: {1}".format(
            self.number_protocols_origin_is2[1],
            str(self.number_protocols_origin_is2[1] / self.total_number_of_protocols)))
        print("--->{0} protocols have more than two origin lang extracted from first language folder files. rate: {1}".format(
            self.number_protocols_origin_is_more[0],
            str(self.number_protocols_origin_is_more[0] / self.total_number_of_protocols)))
        print("--->{0} protocols have more than two origin lang extracted from second language folder files. rate: {1}".format(
            self.number_protocols_origin_is_more[1],
            str(self.number_protocols_origin_is_more[1] / self.total_number_of_protocols)))
        print("--->{0} protocols origin lang is 1. rate: {1}".format(
            self.number_protocols_origin_is1[0],
            str(self.number_protocols_origin_is1[0] / self.total_number_of_protocols)))
        print("--->{0} protocols origin lang is 1. rate: {1}".format(
            self.number_protocols_origin_is1[1],
            str(self.number_protocols_origin_is1[1] / self.total_number_of_protocols)))

        print("{0} protocols origin language was successfully extracted, but did not match any of the two languages. rate {1}".format(
            self.number_protocols_origin_lang_not_any_of_two,
            str(self.number_protocols_origin_lang_not_any_of_two / self.total_number_of_protocols)))
        print("{0} protocols origin language was matching one of the two, however link file could not be obtained. rate {1}". format(
            self.number_of_protocols_not_obtainable_link_file,
            str(self.number_of_protocols_not_obtainable_link_file/self.total_number_of_protocols)))
        print("**** Valid Protocols ****")
        print("{0} protocols origin language was successfully extracted, and matched the first language. rate: {1}".format(
            self.number_of_protocols_originally_first_lang,
            str(self.number_of_protocols_originally_first_lang / self.total_number_of_protocols)))
        print("{0} protocols origin language was successfully extracted, and matched the second language. rate: {1}".format(
            self.number_of_protocols_originally_second_lang,
            str(self.number_of_protocols_originally_second_lang / self.total_number_of_protocols)))
        print("=================================SENTENCES=================================")
        print("Total number of sentences:{0} out of them {1} valid sentences, the rate is {2}".format(
            self.total_number_of_sentences, self.number_of_valid_sentences,
            str(self.number_of_valid_sentences / self.total_number_of_sentences)))
        print("{0} sentences from lang1 files folder were filtered out because they didn't match it. rate {1}".format(
            self.number_of_sentences_not_match_lang1,
            str(self.number_of_sentences_not_match_lang1 / self.total_number_of_sentences)))
        print("{0} sentences were not found in the link file. rate {1}".format(
            self.number_sentences_not_found_in_link_file,
            str(self.number_sentences_not_found_in_link_file / self.total_number_of_sentences)))
        print("{0} sentences that the aligned id extracted from the links file was not found in the 2nd lang protocol {1}".format(
            self.number_of_sentences_aligned_id_from_links_not_found_second_file,
            str(self.number_of_sentences_aligned_id_from_links_not_found_second_file / self.total_number_of_sentences)))
        print("{0} sentences from lang2 files folder were filtered out because they were not in lang2. rate {1}".format(
            self.number_of_sentences_not_match_lang2,
            str(self.number_of_sentences_not_match_lang2 / self.total_number_of_sentences)))
        print("{0} sentences were filtered out due to writing exception. rate {1}".format(
            self.number_of_sentences_writing_exception,
            str(self.number_of_sentences_writing_exception / self.total_number_of_sentences)))
        print("**** Valid Sentences ****")
        print("{0} sentences are originally in the first language. rate {1} out of the valid sentences".format(
            self.number_of_sentences_in_originally_first_lang,
            str(self.number_of_sentences_in_originally_first_lang / self.total_number_of_sentences)))
        print("{0} sentences are originally in second language. rate {1} out of the valid sentences ".format(
            self.number_of_sentences_in_originally_second_lang,
            str(self.number_of_sentences_in_originally_second_lang / self.total_number_of_sentences)))



def find_original_language_in_one_file(xmlRoot, lang, path, idx, statistics):
    for s in xmlRoot.iter('s'):
        if s.text is not None and any(word in s.text for word in ['ORIGINAL', 'Original']):
            try:
                if ':' in s.text:
                    extracted =  s.text.split(':')[1].strip().lower()
                else:
                    extracted = s.text.split(' ')[1].strip().lower()
            except:
                #print("******the original language could not be extracted: " + str(s.text) + " in path: " +  str(path) + " In sentence id: " + s.attrib['id'] )
                continue
            num = 0
            original_lang1_compare = None
            original_language = ""
            for list_lang_tuple in languages_dict[lang]:
                for lang_word in list_lang_tuple[0]:
                    if lang_word in extracted:
                        num += 1
                        original_lang1_compare = list_lang_tuple
                        original_language += lang_word + "-"
            if num == 0:
                #print("In get_origin_language, num==0 extracted:{0} ,Where original_language:{1} ,In path: {2} ,In sentence id:{3}".format(
                #    extracted, lang, str(path), s.attrib['id']))
                statistics.number_protocols_origin_is0[idx] += 1
                return None
            if num == 2:
                #print("In get_origin_language, num==2 extracted:{0} ,Where original_language:{1} in path: {2},In sentence id:{3}".format(
                #    extracted, str(original_language),str(path),s.attrib['id']))
                statistics.number_protocols_origin_is2[idx] += 1
                return None
            if num != 1:
                #print("In get_origin_language, num>2 extracted:{0} ,Where original_language:{1} in path: {2},In sentence id:{3}".format(
                #    extracted, str(original_language),str(path),s.attrib['id']))
                statistics.number_protocols_origin_is_more[idx] += 1
                return None
            statistics.number_protocols_origin_is1[idx] += 1
            #print("In get_origin_language, num===1 extracted:{0} ,Where original_language:{1} in path: {2},In sentence id:{3}".format(
            #    extracted, str(original_language),str(path),s.attrib['id']))
            return original_lang1_compare
    else:
        #print("went over the entire file: " + str(path) + " did not find any Original sentance")
        statistics.number_protocols_not_contain_original_tag[idx] += 1

def get_origin_language(xmlRoot1, lang1, xmlRoot2, lang2, path1, path2, statistics):
    comp1 = find_original_language_in_one_file(xmlRoot1, lang1, path1, 0, statistics)
    comp2 = find_original_language_in_one_file(xmlRoot2, lang2, path2, 1, statistics)
    #print("COMPARING ORIGINAL LAHGUANEG FROM TWO FILES: ", comp1, comp2)
    if comp1 is None and comp2 is None:
        statistics.number_protocols_not_extracted_origin_both += 1
    elif comp1 is None and comp2 is not None:
        statistics.number_protocols_not_extracted_origin_only_first += 1
    elif comp1 is not None and comp2 is None:
        statistics.number_protocols_not_extracted_origin_only_second += 1
    elif (comp1 is not None and comp2 is not None and comp1[1] != comp2[1]):
        statistics.number_protocols_extracted_origin_conflicted += 1
        #print(comp1, comp2, '^^^^', path1, path2)
    elif (comp1 is not None and comp2 is not None and comp1[1] == comp2[1]):
        return comp1[1]
    return None

def get_link_map(lang1, lang2, path):
    link_map = {}
    lnk_folder = [val for val in links if lang1 in val and lang2 in val].pop()
    link_path = glob.glob(os.path.join(language_pair_to_path[lnk_folder], path.split(".xml")[0] + ".lnk"))     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    try:
        links_file = open(link_path.pop(), 'r')
        links_root = ET.parse(links_file).getroot()
        if (lang1+"_"+lang2) == lnk_folder:
            for link in links_root.iter('link'):
                if link.attrib['type'] == '1-1':
                    link_string = link.attrib['xtargets'].split(';')
                    link_map[link_string[0]] = link_string[1]
        elif (lang2+"_"+lang1) == lnk_folder:
            for link in links_root.iter('link'):
                if link.attrib['type'] == '1-1':
                    link_string = link.attrib['xtargets'].split(';')
                    link_map[link_string[1]] = link_string[0]
        else:
            print("***impossible option****")
            return None

        links_file.close()
        return link_map
    except:
        print("Could not open/parse the protocol {0},{1} link file. In path:{2} ".format(lang1,lang2,path) + sys.exc_info())
        return None

def find_file_in_path(path_to_file, idx, stat):
    if os.path.isfile(path_to_file):
        try:
            return ET.parse(path_to_file).getroot()
        except:
            #print("Failed to Parse " + str(path_to_file), sys.exc_info())
            stat.number_protocols_unobtainable_xml_unparseable[idx] += 1
    else:
        stat.number_protocols_unobtainable_file_not_found[idx] += 1
    return None
"""
Returns a dictionary between sentence id to the sentence text
"""
def get_id_to_text(xml_root):
    #TODO: filter out sentences that are not in origin language
    ids = [s.attrib['id'] for s in xml_root.iter('s')]
    text = [(s.text, s.attrib['lang']) for s in xml_root.iter('s')]
    return {x: y for x, y in zip(ids, text)}

def create_output_files(output_folder, lang1, lang2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    first_lang = open(os.path.join(output_folder, lang1 + ".txt"), 'w+', encoding='utf-8')
    second_lang = open(os.path.join(output_folder, lang2 + ".txt"), 'w+', encoding='utf-8')
    index_bilingual = open(os.path.join(output_folder, lang1 + '-' + lang2 + ".txt"), 'w+', encoding='utf-8')
    return first_lang, second_lang, index_bilingual

def build_parallel_corpus(lang1, lang2, output_folder, statistics):
    lang1_file, lang2_file, index_file =  create_output_files(output_folder, lang1, lang2)
    for root, dirs, files in os.walk(lang_to_path[lang1]):
        for file in files:
            if not file.lower().endswith("xml"):
                continue
            statistics.total_number_of_protocols += 1
            first_root = find_file_in_path(os.path.join(root, file), 0, statistics)
            if first_root is None:
                statistics.number_protocols_unobtainable_xml_first_root += 1
                continue
            file_relative_path = os.path.relpath(os.path.join(root, file), start=lang_to_path[lang1])
            second_root = find_file_in_path(os.path.join(lang_to_path[lang2], file_relative_path), 1,  statistics)
            if second_root is None:
                statistics.number_protocols_unobtainable_xml_second_root += 1
                continue

            origin_language = get_origin_language(first_root, lang1, second_root, lang2,
                                                  os.path.join(root, file), os.path.join(lang_to_path[lang2], file_relative_path),
                                                  statistics)
            if origin_language is None:
                statistics.number_protocols_unobtainable_origin_lang += 1
                continue
            if origin_language != lang1 and origin_language != lang2:
                statistics.number_protocols_origin_lang_not_any_of_two += 1
                continue
            second_id_to_text = get_id_to_text(second_root)
            links = get_link_map(lang1, lang2, file_relative_path)
            if links is None:
                statistics.number_of_protocols_not_obtainable_link_file += 1
                continue
            statistics.number_of_valid_protocols += 1
            if origin_language == lang1:
                statistics.number_of_protocols_originally_first_lang += 1
            elif origin_language == lang2:
                statistics.number_of_protocols_originally_second_lang += 1
            else:
                print("2!!!!!!!Unavailable!!!!!!!!!")
            for sentence in first_root.iter('s'):
                statistics.total_number_of_sentences += 1
                if sentence.attrib['lang'] != lang1:
                    #print("A sentence from lang1 folder did not match lang1:" + str(lang1) + ", equals:" + str(sentence.attrib['lang']) + " in file:" + os.path.join(root, file))
                    statistics.number_of_sentences_not_match_lang1 += 1
                    continue
                id = sentence.attrib['id']
                if id not in links.keys():
                    statistics.number_sentences_not_found_in_link_file += 1
                    continue
                aligned_id = links[id]
                # the following if is needed: text originally  from french to english protocol 1990 -> add_1.xml
                # reaching   link dict entry "34:2;32:2" keys exists in french protocol however, does not exist in english protocol
                if (aligned_id not in second_id_to_text.keys()):
                    statistics.number_of_sentences_aligned_id_from_links_not_found_second_file += 1
                    #print('!!!!protocol: {0}, links_id:({1},{2})'.format(os.path.join(root, file),id,aligned_id))
                    continue
                mapped_sentence = second_id_to_text[aligned_id]
                if mapped_sentence[1] != lang2:
                    statistics.number_of_sentences_not_match_lang2 += 1
                    continue
                try:
                    lang1_file.write(sentence.text + '\n')
                    lang2_file.write(mapped_sentence[0] + '\n')
                    index_file.write(origin_language + '\n')
                    if origin_language == lang1:
                        statistics.number_of_sentences_in_originally_first_lang += 1
                    elif origin_language == lang2:
                        statistics.number_of_sentences_in_originally_second_lang +=1
                    else:
                        print("1!!!!!Unavailable!!!!!!")
                    statistics.number_of_valid_sentences += 1
                except:
                    statistics.number_of_sentences_writing_exception += 1
                    print("could not write to file: ", sys.exc_info())
    print('Closing files')
    lang1_file.close()
    lang2_file.close()
    index_file.close()

if __name__ == "__main__":
    for lang_key in ['fr', 'es', 'ru', 'ar', 'zh']:
        print('***********************************************************')
        print(datetime.now().strftime('Started at: %Y-%m-%d %H:%M:%S'))
        stat = CorpusStatistics()
        build_parallel_corpus(lang_key, 'en', result_output_folder + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), stat)
        print("\n=========Results of {0} -> en =====".format(lang_key))
        stat.print_stat()

    #print(datetime.now().strftime('Started at: %Y-%m-%d %H:%M:%S'))
    #stat_es_en = CorpusStatistics()
    #build_parallel_corpus('es', 'en', result_output_folder + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), stat_es_en)
    #print("\n=========Results of es -> en =====")
    #stat_es_en.print_stat()

    #print(datetime.now().strftime('Started at: %Y-%m-%d %H:%M:%S'))
    #stat_ru_en = CorpusStatistics()
    #build_parallel_corpus('ru', 'en', result_output_folder + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), stat_ru_en)
    #print("\n=========Results of ru -> en =====")
    #stat_ru_en.print_stat()

    #print(datetime.now().strftime('Started at: %Y-%m-%d %H:%M:%S'))
    #stat_fr_en = CorpusStatistics()
    #build_parallel_corpus('fr', 'en', result_output_folder + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), stat_fr_en)
    #print("\n=========Results of fr -> en =====")
    #stat_fr_en.print_stat()

    #print(datetime.now().strftime('Started at: %Y-%m-%d %H:%M:%S'))
    #stat_en_fr = CorpusStatistics()
    #build_parallel_corpus('en', 'fr', result_output_folder + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), stat_en_fr)
    #print("\n=========Results of en -> fr =====")
    #stat_en_fr.print_stat()

    print(datetime.now().strftime('Finished at: %Y-%m-%d %H:%M:%S'))


# TODO: Add calls for build_parallel_corpus with the rest of the languages note that ordeer of the languages is important
# TODO: The language order should be the same as the order in the link files
# TODO: For example, the name of the french- english link folder is "fr_en", so the call will be
# TODO build_parallel_corpus('fr', 'en', 'C:\\Users\\Elad\\NLP_pro') and nor build_parallel_corpus('en', 'fr', 'C:\\Users\\Elad\\NLP_pro')
