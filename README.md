# Fetal-Health

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
%matplotlib inline
```

```
fetal = pd.read_csv('./fetal_health.csv')
fetal.head()
```

![image](https://user-images.githubusercontent.com/25868126/121400088-c5fccb00-c974-11eb-8917-4123d3137b97.png)



### Features

 - **baseline value** FHR baseline (beats per minute)
 - **accelerations** Number of accelerations per second
 - **fetal_movement** Number of fetal movements per second
 - **uterine_contractions** Number of uterine contractions per second
 - **light_decelerations** Number of light decelerations per second
 - **severe_decelerations** Number of severe decelerations per second
 - **prolongued_decelerations** Number of prolonged decelerations per second
 - **abnormal_short_term_variability** Percentage of time with abnormal short term variability
 - **mean_value_of_short_term_variability** Mean value of short term variability
 - **percentage_of_time_with_abnormal_long_term_variability** Percentage of time with abnormal long term variability
 - **mean_value_of_long_term_variability** Mean value of long term variability
 - **histogram_width** Width of FHR histogram
 - **histogram_min** Minimum (low frequency) of FHR histogram
 - **histogram_max** Maximum (high frequency) of FHR histogram
 - **histogram_number_of_peaks** Number of histogram peaks
 - **histogram_number_of_zeroes** Number of histogram zeros
 - **histogram_mode** Histogram mode
 - **histogram_mean** Histogram mean
 - **histogram_median** Histogram median
 - **histogram_variance** Histogram variance
 - **histogram_tendency** Histogram tendency


### Target

 - **fetal_health** Tagged as 1 (Normal), 2 (Suspect) and 3 (Pathological)


```
# 1.Normal 2.Suspect 3.Pathological
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'fetal_health', y = 'fetal_movement', data = fetal)
```

![image](https://user-images.githubusercontent.com/25868126/121400278-078d7600-c975-11eb-9342-82b1e3404ae6.png)

```
colours=["#f7b2b0","#8f7198", "#003f5c"]
sns.countplot(data= fetal, x="fetal_health",palette=colours)
```
![image](https://user-images.githubusercontent.com/25868126/121400405-2db31600-c975-11eb-8abe-2ae4004da818.png)


```
#A quick model selection process
#pipelines of models( it is short was to fit and pred)
pipeline_lr=Pipeline([('lr_classifier',LogisticRegression(random_state=42))])

pipeline_dt=Pipeline([ ('dt_classifier',DecisionTreeClassifier(random_state=42))])

pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier())])

pipeline_svc=Pipeline([('sv_classifier',SVC())])

pipeline_sgd=Pipeline([('sgd_classifier',SGDClassifier(penalty=None))])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_svc,pipeline_sgd]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest', 3: "SVC", 4:"SGD"}


# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

#cross validation on accuracy 
cv_results_accuracy = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train, cv=10 )
    cv_results_accuracy.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))
```


**Logistic Regression: 0.895882**

**Decision Tree: 0.913529**

**RandomForest: 0.938824**

**SVC: 0.912941**

**SGD: 0.891765**
