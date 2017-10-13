"""
Read adult dataset: 
only considers attributes whose indices are in $att_INDEX and 
append target attribute with index $target_INDEX at the end. 

'adult.all' file contains the training and testing original datasets from 
http://archive.ics.uci.edu/ml/index.html

#1. age: continuous, range = [17-90]
#2. workclass: 'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay' (7) 
                  (NB: + 'Never-worked' which appears in 7 lines, but all of them contain ? and thus removed ==> nb_catagories = 7) 
#3. final_weight: continuous, range = [13769-1484705]
#4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, 
                  Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. (16)
#5. education_num: continuous, range = [1-16]
#6. marital_status: 'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'. (7)
#7. occupation: 'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving', 
                   'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv', 'Armed-Forces', 
                   'Priv-house-serv'. (14)
#8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. (6)
#9. race: 'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'. (5)
#10. sex: 'Male', 'Female'. (2)
#11. capital_gain: continuous, range = [0-99999]
#12. capital_loss: continuous, range = [0-4356]
#13. hours_per_week: continuous, range = [1-99]
#15. native_country: 'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 
                         'Iran', 'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 
                         'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru', 
                         'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 
                         'Hungary', 'Holand-Netherlands'. (41)
#15. income_level: '<=50K', '>50K' (2)
"""

#!/usr/bin/env python
# coding=utf-8

#Attribute names as ordered in 'data/adult.all' file 
ATT_NAMES = ['age', 'workclass', 'final_weight', 'education',
             'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
             'native_country', 'income_level']
             
#'False' means that the attribute values are continuous or ordinal  
#'True' means that the attribute is categorical 
CATEGORY = [False, True, False, True, False, True, True,  True,  True,  True,  False,  False,  False,  True,  True]

#Attributes to consider: 'education', 'marital_status', 'occupation', 'hours_per_week', 'native_country'
att_INDEX = [3, 5, 6, 12, 13]
# Target attribute: age
target_INDEX = 0


#att_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#target_INDEX = 14


def read_adult_data():
    """
    read adult data from 'data/adult.all'
    """
    data = []
    is_cat = [] #specifies whether an attribute is continuous or categorical

    att_num = len(att_INDEX)
    data_file = open('data/adult-short.all', 'rU')
    for line in data_file:
        #remove '\n' characters
        line = line.strip()
        #remove empty and incomplete lines
        #only 45222 records will remain 
        if len(line) == 0 or '?' in line:
            continue
        #remove spaces
        line = line.replace(' ', '')
        #split according to ','
        temp = line.split(',')
        ltemp = []
        for i in range(att_num):
            index = att_INDEX[i]
            ltemp.append(temp[index])
            is_cat.append(CATEGORY[index])
        ltemp.append(temp[target_INDEX])
        data.append(ltemp)
    return data, is_cat


    
    
    
