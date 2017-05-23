import os
import math
path = 'D:\\GitHub\\NLP_Lab\\local_results_afterFix_link_dict\\Results2017-05-23_04-09-16'

first_lang = open(os.path.join(path,"en.txt"), 'r', encoding='utf-8')
second_lang = open(os.path.join(path, "fr.txt"), 'r', encoding='utf-8')
index_bilingual = open(os.path.join(path, "fr-en.txt"), 'r', encoding='utf-8')


first_lang_subset = open(os.path.join(path,"en_m.txt"), 'w', encoding='utf-8')
second_lang_subset = open(os.path.join(path, "fr_m.txt"), 'w', encoding='utf-8')
index_bilingual_subset = open(os.path.join(path, "fr-en_m.txt"), 'w', encoding='utf-8')

i = 0
subsize = 40000
diff = 6000
c_en = 0
c_fr = 0
for line1 in first_lang:
    line1 = line1.strip()
    line2 = next(second_lang).strip()
    linelink = next(index_bilingual).strip()
    if (c_en + c_fr <= subsize):
        t_c_en = c_en
        t_c_fr = c_fr
        if linelink == "en":
            t_c_en = c_en + 1
        elif linelink == 'fr':
            t_c_fr = c_fr + 1
        else:
            print ('empy line')
            continue
        if math.fabs(t_c_en - t_c_fr) <= diff:
            first_lang_subset.write(line1+'\n')
            second_lang_subset.write(line2+'\n')
            index_bilingual_subset.write(linelink+'\n')
            c_en = t_c_en
            c_fr = t_c_fr
    i+=1
print('^^^')
print(i, c_en, c_fr)


first_lang.close()
second_lang.close()
index_bilingual.close()
first_lang_subset.close()
second_lang_subset.close()
index_bilingual_subset.close()
