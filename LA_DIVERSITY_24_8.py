# -*- coding: utf-8 -*-



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


# # set font
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'

# set the style of the axes and the text color
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'

plt.rcParams['font.family'] = "serif"

%matplotlib inline


# read the datasets

df_jobs_struct =df_jobs_struct=pd.read_csv(r'D:\Documents\placements\ner\jobs__gender_coded.csv')


df_la_ethn = \
pd.read_csv(r"D:\Documents\placements\ner\Job_Applicants_by_Gender_and_Ethnicity.csv"
    )




# convert job class to numeric
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

def _process_salary(salary):
    if salary is np.nan:
        salary = str(0)
    if "-" not in salary:
        return int(salary.replace(',', ''))
    else:
        sal_list = salary.split('-')
        med_salary = (int(sal_list[0].replace(',', '')) + int(sal_list[1].replace(',', ''))) / 2
        return med_salary

df_jobs_struct['ENTRY_SALARY_GEN_MED'] = \
df_jobs_struct['ENTRY_SALARY_GEN'].apply(lambda x: _process_salary(x))

def plot_bar(dataframe, x, y):
    ax = sns.barplot(y= dataframe[x], x = dataframe[y], palette=('Blues_d'))
    
    for i, v in enumerate(dataframe[y]):
        ax.text(v + 3, i + .25, str(v), color='Blue', fontweight='bold')
    
   
%config InlineBackend.figure_format = 'retina'

df_dupli = df_jobs_struct.drop_duplicates(subset='JOB_CLASS_NO', keep="first")
temp_coding = df_dupli['coding'].value_counts().sort_values(
    ascending = False).rename_axis(
    'Gender Coding').reset_index(
    name='Counts')

plot_bar(temp_coding, 'Gender Coding', 'Counts')



%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(10, 8))

temp_masc_words = df_dupli['masculine_words'].str.split(',', expand=True).stack().value_counts().sort_values(
    ascending = False).rename_axis(
    'Masculine words').reset_index(
    name='Counts')

plot_bar(temp_masc_words, 'Masculine words', 'Counts')





%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(10, 8))

temp_masc_words = df_dupli['feminine_words'].str.split(',', expand=True).stack().value_counts().sort_values(
    ascending = False).rename_axis(
    'Feminine words').reset_index(
    name='Counts')

plot_bar(temp_masc_words, 'Feminine words', 'Counts')


temp_sal_coding = df_dupli.groupby('coding')['ENTRY_SALARY_GEN_MED'].mean().round().sort_values(
    ascending = False).rename_axis(
    'Gender Coding').reset_index(
    name='Salary')

plot_bar(temp_sal_coding, 'Gender Coding', 'Salary')



temp_sal_exam_type = df_dupli.groupby('EXAM_TYPE')['ENTRY_SALARY_GEN_MED'].mean().round().sort_values(
    ascending = False).rename_axis(
    'Exam Type').reset_index(
    name='Salary')

plot_bar(temp_sal_exam_type, 'Exam Type', 'Salary')






plt.figure(figsize=(10, 20))

s = df_jobs_struct.set_index('ENTRY_SALARY_GEN_MED').EDUCATION_MAJOR.str.split(',',expand=True).stack().to_frame('Messages').reset_index()
s = s.groupby('Messages')['ENTRY_SALARY_GEN_MED'].median().round().sort_values(
    ascending = False).rename_axis(
    'Education Major').reset_index(
    name='Salary')

plot_bar(s[:50], 'Education Major', 'Salary')


plt.figure(figsize=(10, 15))
temp_sal_sc_type = df_jobs_struct.groupby('SCHOOL_TYPE')['ENTRY_SALARY_GEN_MED'].median().round().sort_values(
    ascending = False).rename_axis(
    'School Type').reset_index(
    name='Salary')

plot_bar(temp_sal_sc_type, 'School Type', 'Salary')




temp_job_tit_sal = df_jobs_struct.groupby('JOB_CLASS_TITLE')['ENTRY_SALARY_GEN_MED'].median().round().sort_values(
    ascending = False).rename_axis(
    'Job Title').reset_index(
    name='Salary')

plot_bar(temp_job_tit_sal[:20], 'Job Title', 'Salary')



plot_bar(temp_job_tit_sal.tail(20)[:19], 'Job Title', 'Salary')






