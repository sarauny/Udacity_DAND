# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

### Task 0: Data Overview 

# read the all data
## Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# data overview
import pandas as pd
import numpy as np
df = pd.DataFrame(data_dict)
df = df.T
number_of_non_poi, number_of_poi = df['poi'].value_counts()
print "number of columns:", len(df.columns)
print "name of the columns:\n", df.columns
print "number of non pois / pois:", number_of_non_poi, '/', number_of_poi


all_features_list = df.columns
nan_index = df.columns
nan_columns = ['non_nan', 'nan', 'nan_poi','nan_non_poi', 'poi_nan_ratio', 'non_poi_nan_ratio']
df_nan = pd.DataFrame(index=nan_index, columns=nan_columns)
df_nan = df_nan.fillna(0)

for i in all_features_list:
    for j in df.index:
        if df[i][j] == 'NaN':
            df_nan['nan'][i]+=1
            if df['poi'][j] == True:
                
                df_nan['nan_poi'][i]+=1
            else:
                df_nan['nan_non_poi'][i]+=1
        else:
            df_nan['non_nan'][i]+=1

# poi_nan_ratio column
for i in all_features_list:
    if df_nan['nan_poi'][i] == 0:
        df_nan['poi_nan_ratio'][i] == 'NaN'
    else:
        df_nan['poi_nan_ratio'][i] = round((df_nan['nan_poi'][i])*100.0/number_of_poi, 2)

# non_poi_nan_ratio column
for i in all_features_list:
    if df_nan['nan_non_poi'][i] == 0:
            df_nan['non_poi_nan_ratio'][i] == 'NaN'
    else:
        df_nan['non_poi_nan_ratio'][i] = round((df_nan['nan_non_poi'][i])*100.0/number_of_non_poi, 2)

df_nan_1 = df_nan[['non_nan', 'nan']].sort_values(by=['nan'], ascending = True)
df_nan_2 = df_nan[['poi_nan_ratio', 'non_poi_nan_ratio']].sort_values(by=['poi_nan_ratio'], ascending = True)


print 'Number of missing values in data'       
print df_nan_1['nan']

print 'Ratio of missing values in POI / non-POIs'     
print df_nan_2


### Task 2: Remove Outliers

import matplotlib.pyplot as plt
%matplotlib inline
plt.rc('figure', figsize=(6,2))

### Task 2: Remove outliers
# plot the salary and the bonus data
features_outliers = ["salary", "bonus"]
data_outliers = featureFormat(data_dict, features_outliers)

for point in data_outliers:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary,bonus)

plt.xlabel("salary")
plt.ylabel("bonus");

# find the key of outlier
for key, value in data_dict.items():
    if value['bonus'] == data_outliers.max():
        print 'max:', key, data_outliers.max()
        
# remove the outlier 'TOTAL'
data_dict.pop('TOTAL', 0 )

# second plot
data_outliers = featureFormat(data_dict, features_outliers)
for point in data_outliers:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary,bonus)
    
plt.xlabel("salary")
plt.ylabel("bonus");

outlier_bonus =5000000
outlier_salary = 500000

both_outlier = []
for key, value in data_dict.items():
    if value['bonus'] == 'NaN':
        pass
    elif value['bonus'] > outlier_bonus:
        if value['salary'] > outlier_salary:
            both_outlier.append(key)
    else:
        pass
print 'people whose both bonus and salary are away from other data points'   
print both_outlier

bonus_outlier = []
for key, value in data_dict.items():
    if value['bonus'] == 'NaN':
        pass
    elif value['bonus'] > outlier_bonus:
        if value['salary'] > outlier_salary:
            pass
        else:
            bonus_outlier.append(key)
    else:
        pass

print 'people whose only bonus is away from other data points'       
print bonus_outlier

salary_outlier = []
for key, value in data_dict.items():
    if value['salary'] == 'NaN':
        pass
    elif value['salary'] > outlier_salary:
        if value['bonus'] > outlier_bonus:
            pass
        else:
            salary_outlier.append(key)
    else:
        pass

print 'people whose only bonus is away from other data points'       
print salary_outlier

# find poi/non-poi of outlier
outlier_list = both_outlier + bonus_outlier + salary_outlier

poi_outlier = []
non_poi_outlier = []
for key in outlier_list:
    poi_value = df.get_value(index=key, col='poi')
    if poi_value == True:
        poi_outlier.append(key)
        print '{:>20}: {}'.format(key, poi_value)
    else:
        non_poi_outlier.append(key)  
        print '{:>20}: {}'.format(key, poi_value)

# remove the non-poi outlier
for key in non_poi_outlier:
    data_dict.pop(key, 0 )

# total number of poi nd non-poi after outlier removal    
df = pd.DataFrame(data_dict)
df = df.T
number_of_non_poi, number_of_poi = df['poi'].value_counts()
print "number of non-pois / pois:", number_of_non_poi, '/', number_of_poi


### Task 4. Try a Variety of Classifiers

# cross validation would be done by StratifiedShuffleSplit, based on f1 score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

def validate(features, labels):
    '''
    Ten-fold cross-validation with stratified sampling.
    '''
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    sss = StratifiedShuffleSplit(labels, n_iter=10)
    for train_index, test_index in sss:
        
        features = np.asarray(features)
        labels = np.asarray(labels)
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        clf.fit(features_train, labels_train)
        y_pred = clf.predict(features_test)
        accuracy_scores.append(accuracy_score(labels_test, y_pred))
        precision_scores.append(precision_score(labels_test, y_pred))
        recall_scores.append(recall_score(labels_test, y_pred))
        f1_scores.append(f1_score(labels_test, y_pred))

    return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)   

### Store to my_dataset for easy export below.
my_dataset = data_dict

# all features initially available and the first feature is 'poi'
features_list = list(all_features_list)

# bring the 'poi' feature first on the list for later data process
def poi_first(features_list):
    poi_index = features_list.index('poi')
    if poi_index == 0:
        pass
    else:
        features_list[poi_index], features_list[0] = features_list[0], features_list[poi_index]

poi_first(features_list)
        
# remove email address from all features
features_list.remove('email_address')

### Extract features and labels from dataset again with the new features_list
from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# split data into train and test data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
    
    
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import precision_score, recall_score, f1_score

# loop over several models to find the best one [1][5]
classifiers = [
    AdaBoostClassifier(),
    GaussianNB(),
    svm.SVC(),
    DecisionTreeClassifier()
    ]

def clf_df(classifiers, features, labels):
    clf_col = ['clf', 'precision', 'recall', 'f1 score']
    clf_df = pd.DataFrame(index=[], columns=clf_col)
    
    for clf in classifiers:
        precision, recall, f1score = validate(features, labels)
        series = pd.Series([clf, precision, recall, f1score], index=clf_df.columns)
        clf_df = clf_df.append(series, ignore_index = True)
    return clf_df

print 'accuracy, precision, reacall and f1 scores of each classifier'
clf_df(classifiers, features, labels)


# from the clf_df result, choose the best algorithm to predict poi / non-poi
classifiers = [
    AdaBoostClassifier(),
    DecisionTreeClassifier()
    ]

# features scaling is not needed with DecisionTreeClassifier() in general [3][4]



### Task 3: Create New Features

### Task 3: Create new feature(s)
from fractions import Fraction

### new feature 1: portion of from/to poi messages within all from/to messages
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    fraction = 0.
    if 'email_address' == 'NaN':
        return 0
    else:
        if all_messages == 0 or all_messages == 'NaN':
            fraction = 0
        if poi_messages == 'NaN':
            fraction = 0
        else:
            fraction = Fraction(poi_messages, all_messages)
    return fraction


for name in my_dataset:

    data_point = my_dataset[name]

    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point['fraction_from_poi'] = fraction_from_poi

    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point['fraction_to_poi'] = fraction_to_poi

# add new features to features_list
features_list.extend(['fraction_from_poi', 'fraction_to_poi'])

### Extract features and labels from dataset again with the new features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# validation of the clf with new features
clf_df(classifiers, features, labels)

clf = DecisionTreeClassifier()


### Task 1: Feature Selection 

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# select features for Decision Tree Classifier using SelectKBest, connecting with Pipeline 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV

# features_list and data for now
# features_list
features_list = list(all_features_list)
features_list.remove('email_address')
features_list.extend(['fraction_from_poi', 'fraction_to_poi'])
poi_first(features_list)

# data 
### Extract features and labels from dataset again with the new features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# kbest with pipeline
n_features = np.arange(1, len(features_list))
kbest = SelectKBest(f_classif)
pipeline = Pipeline([('kbest', kbest), ('DT', DecisionTreeClassifier())])

parameters = {
    'kbest__k':n_features
}

grid_search = GridSearchCV(pipeline, parameters, scoring='f1', cv=10)

grid_search.fit(features, labels)
grid_scores = pd.DataFrame(grid_search.grid_scores_)

print 'grid search score'
pd.set_option('display.width', 100)
print grid_scores

print '\nbest parameters set'
best_k = grid_search.best_params_
print best_k

clf = grid_search
validate(features, labels)

# when k=11 for SelectKBest
kbest = SelectKBest(f_classif, k=11)

# remove labels('poi') from the features_list
features_list_rm_poi = features_list
features_list_rm_poi.remove('poi')

features_df = pd.DataFrame(features, columns=features_list_rm_poi)
labels_df = pd.DataFrame(labels)

kbest.fit(features, labels)
features_new = kbest.transform(features)
kbest_features = features_df.columns[kbest.get_support(indices=True)].tolist()
kbest_scores = kbest.scores_

kbest_features = []
for i in np.argsort(kbest_scores):
    kbest_features.append(features_list_rm_poi[i])

kbest_scores = list(kbest_scores)    
kbest_scores = sorted(kbest_scores, reverse=True)

# print 11 best features and its scores
for i in range(11):
    print '{:>25}: {}'.format(kbest_features[i], kbest_scores[i])

# rewrite the features_list with the best_k best features 
features_list = ['poi'] + kbest_features[:11]


# data with new feature list
### Extract features and labels from dataset again with the new features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 5: Tune Your Classifier

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# [8]


# select features using SelectKBest and GridSearch CV, connecting them with pipeline [3][6][7]

n_features = np.arange(1, len(features_list))
kbest = SelectKBest(f_classif, k=11)
pipeline = Pipeline([('kbest', kbest), ('DT', DecisionTreeClassifier())])

parameters = {
    'DT__criterion':['gini', 'entropy'], 
    'DT__min_samples_split':[2, 4, 6, 8, 10, 20],
    'DT__max_depth':[None, 5, 10, 15, 20],
    'DT__min_samples_leaf':[2, 4, 6, 8, 10, 20]
}

grid_search = GridSearchCV(pipeline, parameters, cv=10)

grid_search.fit(features, labels)
grid_search.grid_scores_

print 'best parameters set'
print grid_search.best_params_

clf = grid_search
print clf


clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=6, min_samples_split=20)
validate(features, labels)


### Task 6: Dump Your Classifier, dataset, and features_list

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from tester import dump_classifier_and_data
dump_classifier_and_data(clf, my_dataset, features_list)


# %load tester.py
#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()

