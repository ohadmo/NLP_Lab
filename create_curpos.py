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

# The root directory of the data. The root directory contains all the language folders and the links folders
root_path = "D:\\GitHub\\NLP Lab\\"

# The language name abbreviation
languages = ["en", "fr", "es", "ar", "ch", "ru"]

# A dictionary matching every language to the path of the folder containing the documents of the language
lang_to_path = {x: os.path.join(root_path, x) for x in languages}

# The name of the links folders
links = ["es_en", "ar-en", "fr_en", "ru-en", "zh-en"]

# A dictionary of each language pair to the link directory
language_pair_to_path = {x: os.path.join(root_path,"links", x) for x in links}     # !!!!!!! remove  links !!!! before running !


def find_file_in_language(lang, path):
    path_to_file = os.path.join(lang_to_path[lang], path)
    if os.path.isfile(path_to_file):
        return open(path_to_file, 'r', encoding='utf-8')
    return None


def get_origin_language(file):
    return "french"


def get_link_map(lang1, lang2, path):
    link_map = {}
    # TODO remember reversing
    # os.path.splitext(path)[0] + '.link'
    link_path = glob.glob(os.path.join(language_pair_to_path[lang1 + "_" + lang2],path.split(".xml")[0] + ".lnk"))
    print("get_link_path: ", link_path)
    # os.rename(os.path.join(language_pair_to_path[lang1 + "_" + lang2], os.path.dirname(path) + os.path.splitext(path)[0] + '.lnk') ,
    #           os.path.join(language_pair_to_path[lang1 + "_" + lang2], os.path.dirname(path) + os.path.splitext(path)[0] + '.xml'))
###    path_to_link_file = os.path.join(language_pair_to_path[lang1 + "_" + lang2], path)
    # path_to_link_file = C:\Users\Elad\NLP_pro\data\en_fr\UNv1.0-TE\en\1990\trans\wp_29\1999\14
    links_file = open(link_path.pop(), 'r')
    links_root = ET.parse(links_file).getroot()
    for link in links_root.iter('link'):
        #print(link.attrib['type'])
        if link.attrib['type'] == '1-1':
            link_string = link.attrib['xtargets'].split(';')
            link_map[link_string[0]] = link_string[1]
    links_file.close()
    return link_map


def get_id_to_text(file):
    tree = ET.parse(file)
    root = tree.getroot()
    ids = [s.attrib['id'] for s in root.iter('s')]
    text = [s.text for s in root.iter('s')]
    return {x: y for x, y in zip(ids, text)}


def build_parallel_corpus(lang1, lang2, output_folder):
    lang1_file = open(os.path.join(output_folder, lang1 + ".txt"), 'w+')
    lang2_file = open(os.path.join(output_folder, lang2 + ".txt"), 'w+')
    index_file = open(os.path.join(output_folder, lang1 + '-' + lang2 +".txt"), 'w+')
    for root, dirs, files in os.walk(lang_to_path[lang1]):
        for file in files:
            if file.lower().endswith("xml"):
                origin_language = get_origin_language(file)
                if origin_language is not None:
                    print("from origin file",root, file)
                    file_relative_path = os.path.relpath(os.path.join(root, file), start=lang_to_path[lang1])
                    second_language_file = find_file_in_language(lang2, file_relative_path)
                    if second_language_file is not None: # in case the desired file doesn't exist in the 2nd language path
                        print("not skipping:", os.path.join(root, file) , "----", lang2+"\\"+file_relative_path)
                        second_id_to_text = get_id_to_text(second_language_file)
                        second_root = ET.parse(os.path.join(root, file)).getroot()
                        links = get_link_map(lang1, lang2, file_relative_path)
                        for sentence in second_root.iter('s'):
                            id = sentence.attrib['id']
                            if id in links.keys():
                                aligned_id = links[id]
                                #the following if is needed: text originally  from french to english protocol 1990 -> add_1.xml
                                #reaching link dict entry "34:2;32:2" keys exists in french protocol however, does not exist in english protocol
                                if (aligned_id in second_id_to_text.keys()):
                                    try:  #TODO: look into encoding problems
                                        lang1_file.write(sentence.text + '\n')
                                        lang2_file.write(second_id_to_text[aligned_id] + '\n')
                                        index_file.write(origin_language + '\n')
                                    except :
                                        print("Error writing to file: ",sys.exc_info())


if __name__ == "__main__":
    build_parallel_corpus('fr', 'en', 'D:\\GitHub\\NLP Lab\\')
#build_parallel_corpus('es', 'en', 'C:\\Users\\Elad\\NLP_pro')
#build_parallel_corpus('ru', 'en', 'C:\\Users\\Elad\\NLP_pro')
#build_parallel_corpus('ar', 'en', 'C:\\Users\\Elad\\NLP_pro')
#build_parallel_corpus('ch', 'en', 'C:\\Users\\Elad\\NLP_pro')

#TODO: Add calls for build_parallel_corpus with the rest of the languages note that ordeer of the languages is important
#TODO: The language order should be the same as the order in the link files
#TODO: For example, the name of the french- english link folder is "fr_en", so the call will be
#TODO build_parallel_corpus('fr', 'en', 'C:\\Users\\Elad\\NLP_pro') and nor build_parallel_corpus('en', 'fr', 'C:\\Users\\Elad\\NLP_pro')
