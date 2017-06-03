#go over the files, for every language file, group all original lines together. divide to chunks
#find all pronouns in the corpus
#count pronouns frequencies.
#data points = chunks, dimension = number of pronouns in corpus. values = frequencies. label = {T, O}
from sklearn import svm
from sklearn.cross_validation import cross_val_score

CHUNK_SIZE = 2000
pronouns = {"he", "her", "hers", "herself", "him", "himself", "i",
"it", "itself", "me", "mine", "myself", "one", "oneself", "ours", "ourselves", "she", "theirs", "them",
"themselves", "they", "us", "we", "you", "yourself"}

def divide_to_chnuks(language_file, label_file, lang):
    original_chunks = []
    original_chunk = []
    translated_chunks = []
    translated_chunk = []
    for line, label in zip(language_file.readlines(), label_file.readlines()):
        tokens = line.split(' ')
        tokens = [x.strip() for x in tokens]
        if label.strip() == lang:
            original_chunk += tokens
            if len(original_chunk) > CHUNK_SIZE:
                original_chunks.append(original_chunk)
                original_chunk = []
        else:
            translated_chunk += tokens
            if len(translated_chunk) > CHUNK_SIZE:
                translated_chunks.append(translated_chunk)
                translated_chunk = []
    original_chunks.append(original_chunk)
    translated_chunks.append(translated_chunk)
    return original_chunks, translated_chunks


def incrementCount(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 0


if __name__ == '__main__':
    english_file = open("C:\\Users\\Elad\\CS\\NLP_Lab\\en_m_tagged.txt")
    label_file = open("C:\\Users\\Elad\\CS\\NLP_Lab\\fr-en_m.txt")
    o, t = divide_to_chnuks(english_file, label_file, 'en')
    corpus_pronouns = set()
    english_file = open("C:\\Users\\Elad\\CS\\NLP_Lab\\en_m_tagged.txt")
    for line in english_file.readlines():
        tokens = [x.strip().lower().split('_')[0] for x in line.split(' ')]
        for token in tokens:
            if token in pronouns:
                corpus_pronouns.add(token)
    data = []
    label = []
    for chunk in o:
        tokens = [x.strip().lower().split('_')[0] for x in chunk]
        pronouns_count = {x : 0 for x in corpus_pronouns}
        for  token in tokens:
            if token in corpus_pronouns:
                pronouns_count[token] += 1
        data.append([x / len(chunk) for x in pronouns_count.values()])
        label.append(1)
    for chunk in t:
        tokens = [x.strip().lower().split('_')[0] for x in chunk]
        pronouns_count = {x: 0 for x in corpus_pronouns}
        for token in tokens:
            if token in corpus_pronouns:
                pronouns_count[token] += 1
        data.append([(x * (CHUNK_SIZE / len(chunk))) for x in pronouns_count.values()])
        label.append(0)
    clf = svm.SVC(kernel='linear').fit(data, label)
    print(clf.score(data, label))
    scores = cross_val_score(clf, data, label, cv=5)
    print(scores)



    # pronounsCount = {}
                # incrementCount(pronounsCount, token)



