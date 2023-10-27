# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:21:56 2023

@author: manas
"""
import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt

%matplotlib inline

def convert_jobs_to_df(
        path="D:/Documents/placements/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/*.txt",
        raw_text_col_name='raw_job_text'):
    
    job_list=[]
    
    files=glob.glob(path)
    for file in files:
        with open(file,'r',errors='replace') as f:
            content=f.read()
            job_list.append(content)
            
        
    return pd.DataFrame({raw_text_col_name:job_list})



def _class_code_apply(text):
    # extracting job class code
    match=re.search('Class Code: (\d+)', text)
    class_code=None
    try:
        class_code=match.group(1)
    except:
        class_code=None
    return class_code

def _open_date_apply(text):
    
    open_date=''
    result=re.search(
        "(Class Code:|Class  Code:)(.*)(ANNUAL SALARY|ANNUALSALARY)",
        text
        )
    shortContent=''
    if result:
        shortContent=result.group(2).strip()
        result=re.search(
            "Open Date:( .*)REVISED",
            shortContent,flags=re.IGNORECASE)
        if result:
            open_date=result.group(1).strip()
        if open_date=='':
            result= re.search(
                "Open Date:(.*)\(Exam",
                shortContent,flags=re.IGNORECASE)
            if result:
                open_date=result.group(1).strip()
        if open_date=='':
            result= re.search(
                "Open Date:(.*)",
                shortContent,flags=re.IGNORECASE)
            if result:
                open_date=result.group(1).strip()
    return open_date

def _exam_type_apply(text):
    
    exam_type = ""
    result= re.search(
        "(Class Code:|Class  Code:)(.*)(ANNUAL SALARY|ANNUALSALARY)",
        text)
    
    shortContent=''
    if result:
        shortContent=result.group(2).strip()
        result= re.search(
            "\(+(.*?)\)", shortContent,flags=re.IGNORECASE)
        if result:
            exam_type=result.group(1).strip()
    return exam_type
       
def _salary_apply(text):
    salary = ''
    salary_notes = ''
    result=re.search(
        "(ANNUAL SALARY|ANNUALSALARY)(.*?)DUTIES", text)
    if result:
        salContent= result.group(2).strip()
        if "NOTE:" in salContent or "NOTES:" in salContent:
            result=re.search(
                "(.*?)(NOTE:|NOTES:)",
                salContent,flags=re.IGNORECASE)
            if result:
                salary=result.group(1).strip()  
            result= re.search(
                "(NOTE:|NOTES:)(.*)",
                salContent,flags=re.IGNORECASE)
            if result:
                salary_notes= result.group(2).strip()
        else:
            salary = salContent
    else:
        result=re.search(
            "(ANNUAL SALARY|ANNUALSALARY)(.*?)REQUIREMENT",
            text,flags=re.IGNORECASE)
        if result:
            salContent= result.group(2).strip()
            if "NOTE:" in salContent or "NOTES:" in salContent:
                result=re.search(
                    "(.*?)(NOTE:|NOTES:)",
                    salContent,flags=re.IGNORECASE)
                if result:
                    salary=result.group(1).strip()  
                result= re.search(
                    "(NOTE:|NOTES:)(.*)",
                    salContent,flags=re.IGNORECASE)
                if result:
                    salary_notes= result.group(2).strip()
            else:
                salary= salContent
    salary_text = "|||||||||||||||".join([salary, salary_notes])
    return salary_text


def _duties_apply(text):
    
    duties=''
    result=duties= re.search("DUTIES(.*?)REQUIREMENT", text)
    if result:
        duties= result.group(1).strip()
    return duties

def _requirements_apply(text):
    """
    Extract entire job requirements section
    """
    req='|'.join(["REQUIREMENT/MIMINUMUM QUALIFICATION",
                  "REQUIREMENT/MINUMUM QUALIFICATION",
                  "REQUIREMENT/MINIMUM QUALIFICATION",
                  "REQUIREMENT/MINIMUM QUALIFICATIONS",
                  "REQUIREMENT/ MINIMUM QUALIFICATION",
                  "REQUIREMENTS/MINUMUM QUALIFICATIONS",
                  "REQUIREMENTS/ MINIMUM QUALIFICATIONS",
                  "REQUIREMENTS/MINIMUM QUALIFICATIONS",
                  "REQUIREMENTS/MINIMUM REQUIREMENTS",
                  "REQUIREMENTS/MINIMUM QUALIFCATIONS",
                  "MINIMUM REQUIREMENTS:",
                  "REQUIREMENTS",
                  "REQUIREMENT"])
    
    result= re.search(f"({req})(.*)(WHERE TO APPLY|HOW TO APPLY)", text)
    requirements=''
    if result:
        requirements = result.group(2).strip()
    return requirements

def _where_to_apply(text):
    
    """
    Extract entire 'WHERE TO APPLY' section
    """
    
    where_to_apply = ''
    result= re.search(
        "(HOW TO APPLY|WHERE TO APPLY)(.*)(APPLICATION DEADLINE|APPLICATION PROCESS)",
        text)
    if result:
        where_to_apply= result.group(2).strip()
    else:
        result= re.search(
            "(HOW TO APPLY|WHERE TO APPLY)(.*)(SELECTION PROCESS|SELELCTION PROCESS)",
            text)
        if result:
            where_to_apply= result.group(2).strip()
    return where_to_apply

def _deadline_apply(text):
    """
    Extract entire deadline section
    """
    
    deadline=''
    result= re.search(
        "(APPLICATION DEADLINE|APPLICATION PROCESS)(.*?)(SELECTION PROCESS|SELELCTION PROCESS)",
        text)
    if result:
        deadline= result.group(2).strip()
    else:
        result= re.search(
            "(APPLICATION DEADLINE|APPLICATION PROCESS)(.*?)(Examination Weight:)",
            text)
        if result:
            deadline= result.group(2).strip()
            
    return deadline

def _selection_process_apply(text):
    
    """
    Extract selectioin process section
    """
    
    selection_process=''
    result=selection_process= re.search(
        "(SELECTION PROCESS|Examination Weight:)(.*)(APPOINTMENT|APPOINTMENT IS SUBJECT TO:)",
        text)
    if result:
        selection_process= result.group(2).strip()
    else:
        result=selection_process= re.search(
            "(SELECTION PROCESS|Examination Weight:)(.*)",
            text)
        if result:
            selection_process= result.group(2).strip()
            
    return selection_process
















def _whole_clean_text(text):
    return text.replace("\n","").replace("\t","").strip()

def pre_processing(dataframe):
    #removing all 1st nrw line characters
    dataframe['raw_job_text']=dataframe['raw_job_text'].apply(
        lambda x: x.lstrip())
    return dataframe

def extract_job_title(dataframe):
   
    dataframe['JOB_CLASS_TITLE'] = dataframe['raw_job_text'].apply(
        lambda x: x.split('\n', 1)[0])
    dataframe['JOB_CLASS_TITLE'] = dataframe['JOB_CLASS_TITLE'].apply(
        lambda x: _whole_clean_text(x))
    return dataframe   

def extract_class_code(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    # find class code
    dataframe['JOB_CLASS_NO'] = temp.apply(lambda x: _class_code_apply(x))
    return dataframe


def extract_open_date(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['OPEN_DATE'] = temp.apply(lambda x: _open_date_apply(x))
    return dataframe


def extract_exam_type(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['TEMP_EXAM_TYPE'] = temp.apply(lambda x: _exam_type_apply(x))
    return dataframe

def extract_salary(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['TEMP_SALARY'] = temp.apply(lambda x: _salary_apply(x))
    return dataframe

def extract_duties(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['JOB_DUTIES'] = temp.apply(lambda x: _duties_apply(x))
    return dataframe

def extract_requirements(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['TEMP_REQUIREMENTS'] = temp.apply(lambda x: _requirements_apply(x))
    return dataframe

def extract_where_to_apply(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['WHERE_TO_APPLY'] = temp.apply(lambda x: _where_to_apply(x))
    return dataframe

def extract_deadline(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['DEADLINE'] = temp.apply(lambda x: _deadline_apply(x))
    return dataframe


def extract_selection_process(dataframe):
    # remove all extra white spaces
    temp = dataframe['raw_job_text'].apply(lambda x: ' '.join(x.split()))
    
    dataframe['SELECTION_PROCESS'] = temp.apply(lambda x: _selection_process_apply(x))
    return dataframe



df_jobs=convert_jobs_to_df()

df_jobs=pre_processing(df_jobs)

df_jobs = extract_job_title(df_jobs) # extract job title

df_jobs = extract_class_code(df_jobs) # extract class code

df_jobs = extract_open_date(df_jobs) # extract open date

df_jobs = extract_exam_type(df_jobs) # extract exam type section

df_jobs = extract_salary(df_jobs) # extract salary section

df_jobs = extract_duties(df_jobs) # extract duties section

df_jobs = extract_requirements(df_jobs) # extract requirements section

df_jobs = extract_where_to_apply(df_jobs) # extract where to apply section

df_jobs = extract_deadline(df_jobs) # extract deadline section

df_jobs = extract_selection_process(df_jobs) # extract selectin pro section

# create a new column containing whole text but clean from new line and tab 
df_jobs['raw_clean_job_text'] = df_jobs['raw_job_text'].apply(
    lambda x: _whole_clean_text(x))


df_jobs.head()         


#df_jobs.to_csv("D:\Documents\placements\jobs.csv")


#=================

def salary(content):   
    try:
        salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?') #match salary
        sal=re.search(salary,content)
        if sal:
            range1=sal.group(1)
            if range1 and '$' not in range1:
                range1='$'+range1
            range2=sal.group(2)
            if range2:
                range2=sal.group(2).replace('to','')
                range2=range2.replace('and','')
            if range1 and range2:
                return f"{range1}-{range2.strip()}"
            elif range1:
                return f"{range1} (flat-rated)"
        else:
            return ''
    except Exception as e:
        return ''


# ENTRY_SALARY_DWP

def salaryDWP(content):
    try:
        result= re.search("(Department of Water and Power is)(.*)", content)
        if result:
            salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?') #match salary
            sal=re.search(salary,result.group(2))
            if sal:
                range1=sal.group(1)
                if range1 and '$' not in range1:
                    range1='$'+range1
                range2=sal.group(2)
                if range2:
                    range2=sal.group(2).replace('to','')
                    range2=range2.replace('and','')
                if range1 and range2:
                    return f"{range1}-{range2.strip()}"
                elif range1:
                    return f"{range1} (flat-rated)"
            else:
                return ''
    except Exception as e:
        return ''


# DRIVERS_LICENSE_REQ

def drivingLicenseReq(content):
    try:
        result= re.search(
            "(.*?)(California driver\'s license|driver\'s license)",
            content)
        
        if result:
            exp=result.group(1).strip()
            exp=' '.join(exp.split()[-10:]).lower()
            if 'may require' in exp:
                return 'P'
            else:
                return 'R'
        else:
            return ''
    except Exception as e:
        return '' 
    


#DRIV_LIC_TYPE

def drivingLicense(content):
    driving_License=[]
    result= re.search(
        "(valid California Class|valid Class|valid California Commercial Class)(.*?)(California driver\'s license|driver\'s license)",
        content)
    if result:
        dl=result.group(2).strip()
        dl=dl.replace("Class","").replace("commercial","").replace("or","").replace("and","")
        if 'A' in dl:
            driving_License.append('A')
        if 'B' in dl:
            driving_License.append('B') 
        if 'C' in dl:
            driving_License.append('C')  
        if 'I' in dl:
            driving_License.append('I')   
        return ','.join(driving_License)
    else:
        return ''
    

#EXAM_TYPE

def examType(content):
 
    exam_type=''
    if 'INTERDEPARTMENTAL PROMOTIONAL AND AN OPEN COMPETITIVE BASIS' in content:
        exam_type='OPEN_INT_PROM' 
    elif 'OPEN COMPETITIVE BASIS' in content:
         exam_type='OPEN'
    elif 'INTERDEPARTMENTAL PROMOTIONAL' or 'INTERDEPARMENTAL PROMOTIONAL' in content:
        exam_type='INT_DEPT_PROM'
    elif 'DEPARTMENTAL PROMOTIONAL' in content:
        exam_type='DEPT_PROM' 
    return exam_type


def split_salary(text):
    
    return text.split('|||||||||||||||')[0]


df_jobs['TEMP_SALARY'] = df_jobs['TEMP_SALARY'].apply(lambda x: split_salary(x))

# extract ENTRY_SALARY_GEN and ENTRY_SALARY_DWP
df_jobs['ENTRY_SALARY_GEN'] = df_jobs['TEMP_SALARY'].apply(lambda x: salary(x))
df_jobs['ENTRY_SALARY_DWP'] = df_jobs['TEMP_SALARY'].apply(lambda x: salaryDWP(x))


# extract DRIVERS_LICENSE_REQ and DRIV_LIC_TYPE
df_jobs['DRIVERS_LICENSE_REQ'] = df_jobs['TEMP_REQUIREMENTS'].apply(
    lambda x: drivingLicenseReq(x))
df_jobs['DRIV_LIC_TYPE'] = df_jobs['raw_clean_job_text'].apply(lambda x: drivingLicense(x))

# extract EXAM_TYPE
df_jobs['EXAM_TYPE'] = df_jobs['raw_clean_job_text'].apply(lambda x: examType(x))

# final
df_jobs[['ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP',
         'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'EXAM_TYPE']].head(10)



#=== NER model===

# function for seperating process notes
# from requirements section
def _seperate_process_notes(text):
    result = re.search('(.*)(PROCESS NOTES|NOTES)(.*)', text)
    if result:
        req = result.group(1)
        process_notes = result.group(3)
    else:
        req = text 
        process_notes = None
    return req, process_notes


def _split_requirements(text):
    req_list = re.split('or \d\.', text)
    return req_list


def split_list_to_rows(dataframe, col_name):
    # here col_name is the name of the column which contains list
    return pd.DataFrame({
          col:np.repeat(dataframe[col].values, dataframe[col_name].str.len())
          for col in dataframe.columns.drop(col_name)}
        ).assign(**{col_name:np.concatenate(dataframe[col_name].values)})[dataframe.columns]


# seperate process notes
df_jobs['REQUIREMENTS'], df_jobs['REQUIREMENTS_PROCESS'] = \
zip(*df_jobs['TEMP_REQUIREMENTS'].apply(lambda x: _seperate_process_notes(x)))

# remove some text
df_jobs['REQUIREMENTS'] = df_jobs['REQUIREMENTS'].str.replace(r'PROCESS', '')
# df_jobs['REQUIREMENTS'].to_csv('./jobs_desc.csv', index=None)


# split requirements and store them as seperate rows
df_jobs['req_list'] = df_jobs['REQUIREMENTS'].apply(lambda x: _split_requirements(x))
df_jobs = split_list_to_rows(df_jobs, 'req_list')


# final
df_jobs[['JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'req_list']].head(20)


def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return training_data



import logging
import json
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

TRAIN_DATA = convert_dataturks_to_spacy("D:/Documents/placements/city_of_la_jobs_ner_label.json")
len(TRAIN_DATA)



nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("D:/Documents/placements/ner/train.spacy") 

nlp = spacy.load(r"D:\Documents\placements\ner\output\model-best")



# function for extracting data field value
def _ner_apply(text):
    
    doc = nlp(text)
    
    return [(ent.text, ent.label_) for ent in doc.ents]



# apply the function and store the result in a new column 
df_jobs['temp_entity'] = df_jobs['req_list'].apply(lambda x: _ner_apply(x))

df_jobs[['req_list', 'temp_entity']].head(10)





import itertools

# process our data field column and seperate each column and store their value in their column
flatter = sorted([list(x) + [idx] for idx, y in enumerate(df_jobs['temp_entity']) 
                  for x in y], key = lambda x: x[1]) 

# Find all of the values that will eventually go in each F column                
for key, group in itertools.groupby(flatter, lambda x: x[1]):
    list_of_vals = [(val, idx) for val, _, idx in group]

    # Add each value at the appropriate index and F column
    for val, idx in list_of_vals:
        df_jobs.loc[idx, key] = val
        
df_jobs['REQUIREMENT_SET_ID'] = df_jobs.groupby('JOB_CLASS_NO').cumcount() + 1















COLUMNS_ORDER = ['JOB_CLASS_TITLE', 'JOB_CLASS_NO', 
                 'REQUIREMENT_SET_ID',
                 'JOB_DUTIES', 'ENTRY_SALARY_GEN',
                 'ENTRY_SALARY_DWP','OPEN_DATE',
                 'EDUCATION_YEARS', 'SCHOOL_TYPE',
                 'EDUCATION_MAJOR', 'DEGREE NAME','EXPERIENCE_LENGTH',
                 'FULL_TIME_PART_TIME',
                 'EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_FUNCTION',
                 'EXP_JOB_COMPANY',
                 'EXP_JOB_CLASS_ALT_RESP',
                 'COURSE_LENGTH', 'COURSE_SUBJECT',
                 'REQUIRED_CERTIFICATE','CERTIFICATE_ISSUED_BY',
                 'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE',
                 'EXAM_TYPE', 'req_list', 'raw_clean_job_text',
                 'REQUIREMENTS']


df_jobs_clean_col=df_jobs[COLUMNS_ORDER]


df_jobs_clean_col = df_jobs_clean_col[df_jobs_clean_col['EDUCATION_YEARS'].str.contains(
    "year", flags=re.IGNORECASE, na=True)]

df_jobs_clean_col = df_jobs_clean_col[df_jobs_clean_col['SCHOOL_TYPE'].str.contains(
    "school|college|university|apprenticeship|G.E.D", flags=re.IGNORECASE, na=True)]

df_jobs_clean_col = df_jobs_clean_col[df_jobs_clean_col['EXPERIENCE_LENGTH'].str.contains(
    "year|month|hour", flags=re.IGNORECASE, na=True)]

df_jobs_clean_col = df_jobs_clean_col[df_jobs_clean_col['FULL_TIME_PART_TIME'].str.contains(
    "time", flags=re.IGNORECASE, na=True)]


df_jobs_clean_col.head(25)






df_jobs_clean_col.to_csv(r"D:\Documents\placements\ner\job_clean.csv",index=None)





