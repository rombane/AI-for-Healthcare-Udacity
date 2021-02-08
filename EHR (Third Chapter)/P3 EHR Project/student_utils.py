import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    
    df = pd.merge(df, ndc_df[['Proprietary Name', 'NDC_Code']] , how="left", left_on='ndc_code', right_on='NDC_Code')
    df['generic_drug_name'] =  df['Proprietary Name']
    df = df.drop('NDC_Code',axis=1)
    df = df.drop('Proprietary Name',axis=1)

    return df

#Question 4
def select_first_encounter(df):
    
    df.sort_values('encounter_id')
    
    first_encounter_line = df.groupby('patient_nbr')['encounter_id'].head(1)
    first_encounter_df =  df[df['encounter_id'].isin(first_encounter_line.values)]
    
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    rand_df = df.sample(frac=1).reset_index(drop=True)    
    train, validation, test = np.split(rand_df,[int(0.6*len(rand_df)), int(0.8*len(rand_df))])
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_vocab=tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file = vocab_file_path,num_oov_buckets=1)
        tf_categorical_feature_column=tf.feature_column.indicator_column(tf_categorical_vocab)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''

    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,\
                                                              default_value=default_value,\
                                                              normalizer_fn=lambda x:normalize_numeric_with_zscore(x,MEAN,STD),\
                                                              dtype=tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    target_column_value = df[col]
    student_binary_prediction = target_column_value.apply(lambda x: 1 if x >= 5 else 0)
    student_binary_prediction=student_binary_prediction.values
    return student_binary_prediction
