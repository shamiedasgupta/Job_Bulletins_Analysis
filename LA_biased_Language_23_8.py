# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 01:23:14 2023

@author: manas
"""

import numpy as np
import pandas as pd
import re



df_jobs_struct=pd.read_csv(r"D:\Documents\placements\ner\job_clean.csv")

feminine_coded_words = [
    "agree","affectionate","child","cheer","collab","commit","communal",
    "compassion","connect","considerate","cooperat","co-operat",
    "depend","emotiona","empath","feel","flatterable","gentle",
    "honest","interpersonal","interdependen","interpersona","inter-personal",
    "inter-dependen","inter-persona","kind","kinship","loyal","modesty",
    "nag","nurtur","pleasant","polite","quiet","respon","sensitiv",
    "submissive","support","sympath","tender","together","trust","understand",
    "warm","whin","enthusias","inclusive","yield","share","sharin"
]

masculine_coded_words = [
    "active","adventurous","aggress","ambitio",
    "analy","assert","athlet","autonom","battle","boast","challeng",
    "champion","compet","confident","courag","decid","decision","decisive",
    "defend","determin","domina","dominant","driven","fearless","fight",
    "force","greedy","head-strong","headstrong","hierarch","hostil",
    "impulsive","independen","individual","intellect","lead","logic",
    "objective","opinion","outspoken","persist","principle","reckless",
    "self-confiden","self-relian","self-sufficien","selfconfiden",
    "selfrelian","selfsufficien","stubborn","superior","unreasonab"
]

hyphenated_coded_words = [
    "co-operat","inter-personal","inter-dependen","inter-persona",
    "self-confiden","self-relian","self-sufficien"
]

possible_codings = (
    "strongly feminine-coded","feminine-coded","neutral",
    "masculine-coded","strongly masculine-coded"
)


class JobAd():
    def __init__(self, ad_text):
        self.ad_text = ad_text
        self.analyse()
        
    def gender_decode(self):
        return self.coding, self.masculine_coded_words, self.feminine_coded_words
        


    def analyse(self):
        word_list = self.clean_up_word_list()
        self.extract_coded_words(word_list)
        self.assess_coding()
        

    def clean_up_word_list(self):
        cleaner_text = ''.join([i if ord(i) < 128 else ' '
            for i in self.ad_text])
        cleaner_text = re.sub("[\\s]", " ", cleaner_text, 0, 0)
        cleaned_word_list = re.sub(u"[\.\t\,“”‘’<>\*\?\!\"\[\]\@\':;\(\)\./&]",
            " ", cleaner_text, 0, 0).split(" ")
        word_list = [word.lower() for word in cleaned_word_list if word != ""]
        return self.de_hyphen_non_coded_words(word_list)

    def de_hyphen_non_coded_words(self, word_list):
        for word in word_list:
            if word.find("-"):
                is_coded_word = False
                for coded_word in hyphenated_coded_words:
                    if word.startswith(coded_word):
                        is_coded_word = True
                if not is_coded_word:
                    word_index = word_list.index(word)
                    word_list.remove(word)
                    split_words = word.split("-")
                    word_list = (word_list[:word_index] + split_words +
                        word_list[word_index:])
        return word_list

    def extract_coded_words(self, advert_word_list):
        words, count = self.find_and_count_coded_words(advert_word_list,
            masculine_coded_words)
        self.masculine_coded_words, self.masculine_word_count = words, count
        words, count = self.find_and_count_coded_words(advert_word_list,
            feminine_coded_words)
        self.feminine_coded_words, self.feminine_word_count = words, count

    def find_and_count_coded_words(self, advert_word_list, gendered_word_list):
        gender_coded_words = list(filter(lambda x: x.lower().startswith(tuple(gendered_word_list)), advert_word_list))
        return (",").join(gender_coded_words), len(gender_coded_words)

    def assess_coding(self):
        coding_score = self.feminine_word_count - self.masculine_word_count
        if coding_score == 0:
            if self.feminine_word_count:
                self.coding = "neutral"
            else:
                self.coding = "empty"
        elif coding_score > 3:
            self.coding = "strongly feminine-coded"
        elif coding_score > 0:
            self.coding = "feminine-coded"
        elif coding_score < -3:
            self.coding = "strongly masculine-coded"
        else:
            self.coding = "masculine-coded"

    def list_words(self):
        if self.masculine_coded_words == "":
            masculine_coded_words = []
        else:
            masculine_coded_words = self.masculine_coded_words.split(",")
        if self.feminine_coded_words == "":
            feminine_coded_words = []
        else:
            feminine_coded_words = self.feminine_coded_words.split(",")
        masculine_coded_words = self.handle_duplicates(masculine_coded_words)
        feminine_coded_words = self.handle_duplicates(feminine_coded_words)
        return masculine_coded_words, feminine_coded_words

    def handle_duplicates(self, word_list):
        d = {}
        l = []
        for item in word_list:
            if item not in d.keys():
                d[item] = 1
            else:
                d[item] += 1
        for key, value in d.items():
            if value == 1:
                l.append(key)
            else:
                l.append("{0} ({1} times)".format(key, value))
        return l




df_jobs_struct['DUTY_REQ'] = df_jobs_struct['JOB_DUTIES'] + df_jobs_struct['REQUIREMENTS']
df_jobs_struct['DUTY_REQ'] = df_jobs_struct['DUTY_REQ'].fillna('No text found')


# apply the class to every of our pandas dataframe
df_jobs_struct['coding'], df_jobs_struct['masculine_words'], df_jobs_struct['feminine_words'] = \
zip(*df_jobs_struct['DUTY_REQ'].apply(lambda x: JobAd(x).gender_decode()))



# finally print extracted coding for each job rows
df_jobs_struct[['JOB_CLASS_TITLE', 'coding', 'masculine_words', 'feminine_words']].head(25)



# save the dataframe for future use
df_jobs_struct.to_csv(r'D:\Documents\placements\ner\jobs__gender_coded.csv', index=False)



def _process_salary(salary):
    if salary is np.nan:
        salary = str(0)
    if "-" not in salary:
        return int(salary.replace(',', ''))
    else:
        sal_list = salary.split('-')
        med_salary = (int(sal_list[0].replace(',', '')) + int(sal_list[1].replace(',', ''))) / 2
        return med_salary

df_la_ethn = \
pd.read_csv(r"D:\Documents\placements\ner\Job_Applicants_by_Gender_and_Ethnicity.csv"
    )

# see what we have got # 
df_la_ethn.head()

df_jobs_struct['JOB_CLASS_NO'] = pd.to_numeric(
    df_jobs_struct['JOB_CLASS_NO'])
df_la_ethn['Job Number'] = df_la_ethn['Job Number'].str.extract('(\d+)')
df_la_ethn['Job Number'] = pd.to_numeric(df_la_ethn['Job Number'])


df_jobs_struct['ENTRY_SALARY_GEN'] = \
df_jobs_struct['ENTRY_SALARY_GEN'].replace({'\$':''}, regex = True)

df_jobs_struct['ENTRY_SALARY_GEN'] = \
df_jobs_struct['ENTRY_SALARY_GEN'].replace({'\(flat-rated\)':''}, regex = True)

df_jobs_struct['ENTRY_SALARY_GEN'] = \
df_jobs_struct['ENTRY_SALARY_GEN'].replace({'nan':''}, regex = True)



df_jobs_struct['ENTRY_SALARY_GEN_MED'] = \
df_jobs_struct['ENTRY_SALARY_GEN'].apply(lambda x: _process_salary(x))


###### Bins the salary for our machine learning model ###########
bins = [0, 50000, 75000, 90000, 100000, 120000, 150000, 200000]
df_jobs_struct['ENTRY_SALARY_GEN_MED_BIN'] = pd.cut(
    df_jobs_struct['ENTRY_SALARY_GEN_MED'], bins)

###############################
# Finally merge our two dataset on job class number
##############################
df_jobs_struct_merge = df_jobs_struct.merge(
    df_la_ethn, left_on='JOB_CLASS_NO', right_on='Job Number')
df_jobs_struct_merge.head()


#===============



import eli5
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier



df_jobs_struct_merge['female_perc'] = (df_jobs_struct_merge['Female'] / df_jobs_struct_merge['Apps Received']) *100
df_jobs_struct_merge['male_perc'] = (df_jobs_struct_merge['Male'] / df_jobs_struct_merge['Apps Received']) *100

conditions = [
    (df_jobs_struct_merge['female_perc'] >= 50),
    (df_jobs_struct_merge['male_perc'] >= 50)]
choices = [0, 1]
df_jobs_struct_merge['gender_code'] = np.select(conditions, choices, default=1)


# dropping unwanted columns

DROP_COLUMNS = ['raw_job_text', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'OPEN_DATE',
                'TEMP_EXAM_TYPE', 'TEMP_SALARY', 'TEMP_REQUIREMENTS', 'WHERE_TO_APPLY',
                'DEADLINE', 'SELECTION_PROCESS', 'raw_clean_job_text', 'REQUIREMENTS_PROCESS',
                'req_list', 'temp_entity', 'Fiscal Year', 'Job Number', 'Job Description',
                'Apps Received', 'Female', 'Male', 'Unknown_Gender', 'ENTRY_SALARY_GEN',
                'ENTRY_SALARY_DWP', 'REQUIREMENT_SET_ID',
                'Black', 'Hispanic', 'Asian', 'Caucasian', 'American Indian/ Alaskan Native',
                'Filipino', 'Unknown_Ethnicity', 'ENTRY_SALARY_GEN_MED']

# y_only = df_jobs_struct_merge[['Female', 'Male']]

X_only = df_jobs_struct_merge


# replae nan values
X_only['JOB_DUTIES'] = X_only['JOB_DUTIES'].fillna('Not Found')
X_only['DRIVERS_LICENSE_REQ'] = X_only['DRIVERS_LICENSE_REQ'].fillna('Not Found')
X_only['REQUIREMENTS'] = X_only['REQUIREMENTS'].fillna('Not Found')
X_only['EDUCATION_MAJOR'] = X_only['EDUCATION_MAJOR'].fillna('Not Found')
X_only['EXP_JOB_CLASS_FUNCTION'] = X_only['EXP_JOB_CLASS_FUNCTION'].fillna('Not Found')
X_only['EXP_JOB_CLASS_TITLE'] = X_only['EXP_JOB_CLASS_TITLE'].fillna('Not Found')



default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(X_only.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('JOB_DUTIES', TfidfVectorizer(
        stop_words='english',
        preprocessor=build_preprocessor('JOB_DUTIES'))),
    ('REQUIREMENTS', TfidfVectorizer(
        stop_words='english',
        preprocessor=build_preprocessor('REQUIREMENTS'))),
    ('EXP_JOB_CLASS_FUNCTION', TfidfVectorizer(
        preprocessor=build_preprocessor('EXP_JOB_CLASS_FUNCTION')))
])
X_train = vectorizer.fit_transform(X_only.values)

df_jobs_struct_merge['gender_code'] = df_jobs_struct_merge['gender_code'].fillna(1)

y_train = df_jobs_struct_merge['gender_code']
X_train


# spilt our training dataset into training and test set
train_x, test_x, train_y, test_y = train_test_split(
    X_train, y_train,test_size=0.1, random_state = 2019)

# declare our model
model = GradientBoostingClassifier(random_state = 20199, n_estimators = 200)

# train our model
model.fit(train_x, train_y)

# evaluate our model performance
y_pred_valid = model.predict(test_x)
acc = accuracy_score(test_y, y_pred_valid)
print(f'valid acc: {acc:.5f}')

eli5.show_weights(model, vec=vectorizer)



# Store feature weights in an object
html_obj = eli5.show_weights(model, vec=vectorizer)


with open('D:\\Documents\\placements\\ner\\elii.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))


url = r'D:\\Documents\\placements\\ner\\elii.htm'
webbrowser.open(url, new=2)

import webbrowser


y_train.values[15]




