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
languages = ["en", "fr", "es", "ar", "ch", "ru"]

# A dictionary matching every language to the path of the folder containing the documents of the language
lang_to_path = {x: os.path.join(root_path, x) for x in languages}

# The name of the links folders
links = ["es_en", "ar-en", "fr_en", "ru-en", "zh-en"]

# A dictionary of each language pair to the link directory
language_pair_to_path = {x: os.path.join(root_path, "links", x) for x in
                         links}  # !!!!!!! remove  links !!!! before running !


languages_dict = {'fr' : [(['anglais'],'en'),
                          (['français', 'francais'],'fr'),
                          (['espagnol', 'espâgnol'],'es'),
                          (['russe'],'ru'),
                          (['chinois'],'zh'),
                          (['arabe'], 'ar')],
                  'en' : [(['english'],'en'),
                          (['french'],'fr'),
                          (['spanish'],'es'),
                          (['russian'], 'ru'),
                          (['chinese'], 'zh'),
                          (['arabic'], 'ar')]}

class CorpusStatistics:
    def __init__(self):
        self.total_number_of_protocols = 0
        self.number_of_valid_protocols = 0
        self.number_of_protocols_originally_bilingual = 0
        self.total_number_of_sentences = 0
        self.number_of_valid_sentences = 0


def find_original_language_in_one_file(xmlRoot, lang, path):
    for s in xmlRoot.iter('s'):
        if s.text is not None and any(word in s.text for word in ['ORIGINAL', 'Original']):
            try:
                if ':' in s.text:
                    extracted =  s.text.split(':')[1].strip().lower()
                else:
                    extracted = s.text.split(' ')[1].strip().lower()
            except:
                print("******the original language could not be found: " + str(s.text) + "  " +  str(path))
                continue
            num = 0
            original_lang1_compare = None
            original_language = ""
            for list_lang_tuple in languages_dict[lang]:
                for lang_word in list_lang_tuple[0]:
                    if lang_word in extracted:
                        num += 1
                        original_lang1_compare = list_lang_tuple
                        original_language += " " + lang_word
            if num == 0:
                print("In get_origin_language, num==0 extracted:{0} where original_language:{1} in path:{2}".format(extracted, str(original_language),str(path)))
                return None
            if num == 2:
                #print("In get_origin_language, num==2 extracted:{0} where original_language:{1} in path:{2}".format(extracted, str(original_language),str(path)))
                return None
            if num != 1:
                #print("!!!! ", num, str(extracted), str(original_language), path)
                return None
            return original_lang1_compare
    else:
        #print("went over the entire file: " + str(path) + " did not find any Original sentance")
        pass

def get_origin_language(xmlRoot1, lang1, xmlRoot2, lang2, path1, path2):
    comp1 = find_original_language_in_one_file(xmlRoot1, lang1, path1)
    comp2 = find_original_language_in_one_file(xmlRoot2, lang2, path2)
    #print("COMPARING ORIGINAL LAHGUANEG FROM TWO FILES: ", comp1, comp2)
    if (comp1 is not None and comp2 is not None and comp1[1] == comp2[1]):
        return comp1[1]
    return None

def get_link_map(lang1, lang2, path):
    link_map = {}
    # TODO remember reversing
    link_path = glob.glob(os.path.join(language_pair_to_path[lang1 + "_" + lang2], path.split(".xml")[0] + ".lnk"))
    links_file = open(link_path.pop(), 'r')
    links_root = ET.parse(links_file).getroot()
    for link in links_root.iter('link'):
        if link.attrib['type'] == '1-1':
            link_string = link.attrib['xtargets'].split(';')
            link_map[link_string[0]] = link_string[1]
    links_file.close()
    return link_map

def find_file_in_path(path_to_file):
    if os.path.isfile(path_to_file):
        try:
            return ET.parse(path_to_file).getroot()
        except:
            print("Failed to Parse " + str(path_to_file), sys.exc_info())
    return None
"""
Returns a dictionary between sentence id to the sentence text
"""
def get_id_to_text(xml_root):
    ids = [s.attrib['id'] for s in xml_root.iter('s')]
    text = [s.text for s in xml_root.iter('s')]
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
            first_root = find_file_in_path(os.path.join(root, file))
            if first_root is None:
                continue
            file_relative_path = os.path.relpath(os.path.join(root, file), start=lang_to_path[lang1])
            second_root = find_file_in_path(os.path.join(lang_to_path[lang2], file_relative_path))
            if second_root is None: # in case the desired file doesn't exist in the 2nd language path
                continue
            origin_language = get_origin_language(first_root, lang1, second_root, lang2, os.path.join(root, file), os.path.join(lang_to_path[lang2], file_relative_path))
            if origin_language is None:
                continue
            second_id_to_text = get_id_to_text(second_root)
            if second_id_to_text is None:
                continue
            links = get_link_map(lang1, lang2, file_relative_path)
            statistics.number_of_valid_protocols += 1
            for sentence in first_root.iter('s'):
                statistics.total_number_of_sentences += 1
                id = sentence.attrib['id']
                if id in links.keys():
                    aligned_id = links[id]
                    # the following if is needed: text originally  from french to english protocol 1990 -> add_1.xml
                    # reaching   link dict entry "34:2;32:2" keys exists in french protocol however, does not exist in english protocol
                    if (aligned_id in second_id_to_text.keys()):
                        try:
                            lang1_file.write(sentence.text + '\n')
                            lang2_file.write(second_id_to_text[aligned_id] + '\n')
                            index_file.write(origin_language + '\n')
                            statistics.number_of_valid_sentences += 1
                        except:
                            print("Error writing to file: ", sys.exc_info())


if __name__ == "__main__":
    print(datetime.now().strftime('Started at: %Y-%m-%d %H:%M:%S'))
    stat_fr_en = CorpusStatistics()
    build_parallel_corpus('fr', 'en', result_output_folder, stat_fr_en)
    print(datetime.now().strftime('Finished at: %Y-%m-%d %H:%M:%S'))
    print("Total number of protocols:{0} out of them {1} valid protocols, the rate is {2}".format(
        stat_fr_en.total_number_of_protocols, stat_fr_en.number_of_valid_protocols,
        str(stat_fr_en.number_of_valid_protocols / stat_fr_en.total_number_of_protocols)))
    print("Total number of sentences:{0} out of them {1} valid sentences, the rate is {2}".format(
        stat_fr_en.total_number_of_sentences, stat_fr_en.number_of_valid_sentences,
        str(stat_fr_en.number_of_valid_sentences/stat_fr_en.total_number_of_sentences)))


# TODO: Add calls for build_parallel_corpus with the rest of the languages note that ordeer of the languages is important
# TODO: The language order should be the same as the order in the link files
# TODO: For example, the name of the french- english link folder is "fr_en", so the call will be
# TODO build_parallel_corpus('fr', 'en', 'C:\\Users\\Elad\\NLP_pro') and nor build_parallel_corpus('en', 'fr', 'C:\\Users\\Elad\\NLP_pro')
