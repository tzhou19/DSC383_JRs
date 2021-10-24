import pandas as pd
import numpy as np
import math

decision_list = ['year', 'Completed', 'School_Decision', 'student_Decision',
             'final_Decision', 'Been_On_Waitinglist']
gre_list = ['GRE Verbal', 'GRE Verbal Percentile', 
              'GRE Quantitative', 'GRE Quantitative Percentile', 'GRE Analytical Writing', 
             'GRE Analytical Writing Percentile', 'GRE Verified']
gpa_list = ['Overall_GPA']
recommender_list = ['avg_score_round', 'Number_Recommender']
graduation_list = ['Graduation_Year', 'Graduation_Country']
year_after_list = ['Year_after_graduation', 'Just_after_graduation']
job_list = ['job1_duration', 'job2_duration', 'job3_duration', 'max_job_duration']

def school_cleaned(text):
    text = str(text)
    if text == 'nan':
        return np.nan
    else:
        text_list = text.split('/')
        return text_list[0]
    
def student_cleaned(text):
    text = str(text)
    if text == 'nan':
        return np.nan
    else:
        text_list = text.split('/')
        if len(text_list) == 1:
            return np.nan
        else:
            return text_list[1]
        
def final_cleaned(text):
    text = str(text)
    if text == 'nan':
        return np.nan
    else:
        text_list = text.split('/')
        if len(text_list) == 2:
            return text_list[1]
        elif len(text_list) == 3:
            return text_list[-1]

def extract_year(text):
    text = str(text)
    result = ''
    for char in text:
        if char.isnumeric() == True:
            result += char
    return result

def GRE_cleaned(df_raw):
    df_gre = df_raw.copy()
    f_nan = df_gre['GRE Verbal'][0]
    f_nat = df_gre['GRE Test Date'][0]
    df_gre = df_gre.replace(f_nan, np.nan)
    df_gre = df_gre.replace(f_nat, np.nan)

    df_result = df_gre.copy()
    df_result['GRE Take or Not'] = np.where(df_result['GRE Verbal'].isna(), 0, 1)
    return df_result

def tofel_cleaned(df_raw):
    f_nan = df_toefl['TOEFL Total'][0]
    df_toefl = df_toefl.replace(f_nan, np.nan)
    
    df_toefl['TOEFL Percentile'] = df_toefl['TOEFL Total'] / 120.0
    df_toefl['IELTS Percentile'] = df_toefl['IELTS Total'] / 9.0
    df_toefl['language_total'] = np.where(df_toefl['TOEFL Percentile'].isna(), 
                                      df_toefl['IELTS Percentile'], df_toefl['TOEFL Percentile'])
    return df_toefl

def gpa_cleaned(df_raw):
    df_gpa = df_raw.copy()
    f_nan = df_gpa['Institution 1 GPA (4.0 Scale)'][0]
    df_gpa = df_gpa.replace(f_nan, np.nan)
    df_gpa['Overall_GPA'] = np.where(df_gpa['Institution 4 GPA (4.0 Scale)'].isna() == False, 
         (df_gpa['Institution 1 GPA (4.0 Scale)'] + df_gpa['Institution 2 GPA (4.0 Scale)'] 
                         + df_gpa['Institution 3 GPA (4.0 Scale)'] + df_gpa['Institution 4 GPA (4.0 Scale)'])/4,
         np.where(df_gpa['Institution 3 GPA (4.0 Scale)'].isna() == False, 
                  (df_gpa['Institution 1 GPA (4.0 Scale)'] + df_gpa['Institution 2 GPA (4.0 Scale)'] 
                         + df_gpa['Institution 3 GPA (4.0 Scale)'])/3,
         np.where(df_gpa['Institution 2 GPA (4.0 Scale)'].isna() == False, 
                  (df_gpa['Institution 1 GPA (4.0 Scale)'] + df_gpa['Institution 2 GPA (4.0 Scale)'])/2, 
         np.where(df_gpa['Institution 1 GPA (4.0 Scale)'].isna() == False, 
                  df_gpa['Institution 1 GPA (4.0 Scale)'], np.nan))))
    return df_gpa


def clean_recommendor(df_recom):
    df_recom['Number_Recommender'] = np.where(df_recom['Recommender 4 Title'].isna() == False, 4,
                                np.where(df_recom['Recommender 3 Title'].isna() == False, 3,
                                np.where(df_recom['Recommender 2 Title'].isna() == False, 2,
                                np.where(df_recom['Recommender 1 Title'].isna() == False, 1, 0))))
    df_recom['r1_score'] = np.where(df_recom['Recommender 1 Rating'] == 'Among the very best', 5,
                                   np.where(df_recom['Recommender 1 Rating'] == 'Top 5%', 4, 
                                           np.where(df_recom['Recommender 1 Rating'] == 'Top 10%', 3,
                                                   np.where(df_recom['Recommender 1 Rating'] == 'Top Quarter', 2,
                                                           np.where(df_recom['Recommender 1 Rating'] == 'Average', 1, 0)))))
    df_recom['r2_score'] = np.where(df_recom['Recommender 2 Rating'] == 'Among the very best', 5,
                                   np.where(df_recom['Recommender 2 Rating'] == 'Top 5%', 4, 
                                           np.where(df_recom['Recommender 2 Rating'] == 'Top 10%', 3,
                                                   np.where(df_recom['Recommender 2 Rating'] == 'Top Quarter', 2,
                                                           np.where(df_recom['Recommender 2 Rating'] == 'Average', 1, 0)))))
    df_recom['r3_score'] = np.where(df_recom['Recommender 3 Rating'] == 'Among the very best', 5,
                                   np.where(df_recom['Recommender 3 Rating'] == 'Top 5%', 4, 
                                           np.where(df_recom['Recommender 3 Rating'] == 'Top 10%', 3,
                                                   np.where(df_recom['Recommender 3 Rating'] == 'Top Quarter', 2,
                                                           np.where(df_recom['Recommender 3 Rating'] == 'Average', 1, 0)))))
    df_recom['r4_score'] = np.where(df_recom['Recommender 4 Rating'] == 'Among the very best', 5,
                                   np.where(df_recom['Recommender 4 Rating'] == 'Top 5%', 4, 
                                           np.where(df_recom['Recommender 4 Rating'] == 'Top 10%', 3,
                                                   np.where(df_recom['Recommender 4 Rating'] == 'Top Quarter', 2,
                                                           np.where(df_recom['Recommender 4 Rating'] == 'Average', 1, 0)))))
    df_recom['avg_score'] = np.where(df_recom['Number_Recommender'] == 4, 
                                     (df_recom['r1_score']+df_recom['r2_score']+df_recom['r3_score']+df_recom['r4_score'])/4, 
                                    np.where(df_recom['Number_Recommender'] == 3, 
                                         (df_recom['r1_score']+df_recom['r2_score']+df_recom['r3_score'])/3,
                                            np.where(df_recom['Number_Recommender'] == 2, 
                                                 (df_recom['r1_score']+df_recom['r2_score'])/2,
                                                    np.where(df_recom['Number_Recommender'] == 1, 
                                                         df_recom['r1_score'], 0))))
    df_recom['avg_score_round'] = df_recom['avg_score'].round()
    return df_recom


def graduation_year_country(df_raw):
    df_gy = df_raw.copy()
    df_gy['Graduation_Year'] = df_gy['Institution 1 Date Conferred'].dt.year
    list_raw = df_gy['Institution 1 Location']
    list_country = []
    for ele in list_raw:
        ele = str(ele)
        if len(ele) < 3:
            list_country.append('US')
        else:
            list_country.append(ele)
    df_gy['Graduation_Country'] = list_country
    return df_gy


def year_after(df_raw):
    df_year = df_raw.copy()
    df_year['year'] = pd.to_datetime(df_year['year'], format='%Y', errors='coerce')
    df_year['year'] = df_year['year'].dt.year
    df_year['Year_after_graduation'] = df_year['year'] - df_year['Graduation_Year']
    df_year['Just_after_graduation'] = np.where(df_year['Year_after_graduation'].isna(), np.nan, 
                                                np.where(df_year['Year_after_graduation'] < 2, 1, 0))
    return df_year


def job_cleaned(df_raw):
    df_job = df_raw.copy()
    df_job['Number_job'] = np.where(df_job['Job 3 Organization'].isna() == False, 3,
                        np.where(df_job['Job 2 Organization'].isna() == False, 2,
                        np.where(df_job['Job 1 Organization'].isna() == False, 1, 0)))
    df_job['job1_duration'] = (df_job['Job 1 To'] - df_job['Job 1 From']).dt.days
    df_job['job2_duration'] = (df_job['Job 2 To'] - df_job['Job 2 From']).dt.days
    df_job['job3_duration'] = (df_job['Job 3 To'] - df_job['Job 3 From']).dt.days
    df_job['max_job_duration'] = round((df_job[['job1_duration', 'job2_duration', 'job3_duration']].max(axis = 1))/30, 0)
    df_job[['Ref','Job 1 To', 'Job 1 From', 'job1_duration', 'Job 2 To', 'Job 2 From', 'job2_duration', 
            'Job 3 To', 'Job 3 From', 'job3_duration', 'max_job_duration']]
    return df_job
