# AppliedStatistics

## Industry - Health Insurance

## Concept - Applied Statistics

This project used Hypothesis Testing and Visualization to leverage customer's health information like smoking habits, bmi, age, and gender for checking statistical evidence to make valuable decisions of insurance business like charges for health insurance.

## Skills and Tools
Hypothesis Testing, Data visualisation, statistical Inference


#from google.colab import files
#files.upload()

### 1. Import the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
sns.set(color_codes=True)

import warnings
warnings.simplefilter(action='ignore')

from scipy.stats import norm
import scipy.stats as stats
from math import *

pd.set_option('display.max_columns',None )
pd.set_option('display.width', 1000)

import copy
from sklearn.preprocessing import LabelEncoder

### 2. Read the data as a data frame

#Importing data from csv to data frame
df=pd.read_csv('insurance (2).csv')

### 3a. Shape of data


print("Shape of data:",df.shape)
print("Available dataframe shape:\nTotal Columns:",df.shape[1],"\nTotal Rows:",df.shape[0])

#3M - Mean Median Mode

 def get3M( data):
        print("Mean:",np.mean(data))
        print("Meadian:",np.median(data))
        print("Mode:",stats.mode(data)[0])
        print("Mode:",data.mode())
        
get3M(df.age)
print('''Mean and meadian are close to equal but mode is way below mean/median, 
showing highest peak is towards lower range of data set.
 mean>median - slightly Positive or rt skewed''')

       
get3M(df.bmi)
print('''Mean and meadian are close to equal but mode is slightly more than mean/median, 
showing highest peak is close to middle range of data set.
 mean>median - slightly Positive or rt skewed''')

get3M(df.charges)
print('''Mean and meadian are close to equal but mode is very less than mean/median, 
showing highest peak is towards low range of data set.
 mean>median - highly Positive or rt skewed''')

print("Mean:\n",df.mean())
print("\nMedian:\n",df.median())
print("\nMod:\n",df.mode())




### 3b. Data Type of each Attribute

print("Attribute Details:")
print(df.info())
print('''\nColumns are having expected data types.
All are having non null values and of same count.

BMI and Charges are Float having continuous data,
Age and Children are int as discrete numerical values, No. of Children is a numerical value but considered as categorical
Sex, Smooker and Region are object(contains string) categorical data
''')

### Analyzing Data present

print("Top 10 rows:\n",df.head(10))
print("\nBottom 10 rows:\n",df.tail(10))
print("\nSample Values looks widely distributed across all columns")

### 3c. Checking presence of missing values

print(df.isna().sum())
print()
print(df.isnull().sum())
print("\nNo nan/null data present, validated same using unique value and value counts as well")

print(df.sex.unique())
df.sex.value_counts()

print(df.children.unique())
df.children.value_counts()

print(df.smoker.unique())
df.smoker.value_counts()

print(df.region.unique())
df.region.value_counts()

print(df.bmi.unique())
df.bmi.value_counts()

print(df.charges.unique())
df.charges.value_counts()

df.age.unique()

### 3d. 5 point summary of numerical attributes

print(df.describe(include='all'))
print("\n\n")
print(df.describe(include='all',percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]))
print('''
Categorical Values Sex, Smoker, Region seems to have expected variation of unique data.
For numerical continuous Variable
Age: 
- min and max age are from range of 18-64
- Q1 - 27
- Q2 - 39
- Q3 - 51
- mean and 50 percentile are very close giving a idea that majority of insurance hoalders are close to mean age of 39.
- mean and median are similar 39.2 -- 39 - suggests distribution is not too skewed
- std deviation is quite high showing high variation

BMI:
- min and max bmi are 15.96 -- 53.13
- Q1 - 26
- Q2 - 30
- Q3 - 34
- mean and 50% are very close around 30
- mean and median are similar 30.6 -- 30.4 - suggests distribution is not skewed
- considering 20%-80% from level 25.32--35.86, seems a lesser gap for 60% of the data consideiring std deviation of 6.09 which leads to possibiliteis of high outliers
- Std deviation is low

Charges:
- min and max charges are  1121.87 -- 63770.42
- Q1 - 4740.287150
- Q2 - 9382.033000
- Q3 - 16639.912515
- mean and 50% are 113270.42 -- 9382.03
- mean is higher then meadian suggests distribution that is skewed to the right
- 90%, 95% and max values are having a huge gap suggesting outliers presence towards higher charges
- Std deviation is high
''')

### 3e. Distribution of ‘bmi’, ‘age’ and ‘charges’ columns

plt.figure(figsize= (20,15))
plt.subplot(3,3,1)
plt.hist(df.age,bins=100)
plt.xlabel("AGE")

plt.subplot(3,3,2)
plt.hist(df.bmi,bins=100)
plt.xlabel("BMI")

plt.subplot(3,3,3)
plt.hist(df.charges,bins=100)
plt.xlabel("CHARGES")

plt.show();

sns.distplot(df.age,bins=50);
plt.show()
print('''
Distribution shows there is high frequence for age 18-19
For rest of the age groups the distribution is kind of flat line i.e. more or less distributed evenly
''')

sns.distplot(df.bmi,bins=100);
plt.show()
print('''
Distribution seems a little skewed towards right, few outliers towards rt side of the distribution
major peaks or frequencies are between 25-35
''')

sns.distplot(df.charges,bins=100);
plt.show()
print('''
Distribution is highly skewed towards right
Highest peaks or frequencies are under 10K
Major Outliers towards higher charges
''')

### 3f. Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns

df.skew()

# Measure of skewness
print("Age \nSkweness : ", stats.skew(df.loc[:,'age']))
print("Kurtosis : ", stats.kurtosis(df.loc[:,'age']))
print("\nBMI\nSkweness : ", stats.skew(df.loc[:,'bmi']))
print("Kurtosis : ", stats.kurtosis(df.loc[:,'bmi']))
print("\nCharges\nSkweness : ", stats.skew(df.loc[:,'charges']))
print("Kurtosis : ", stats.kurtosis(df.loc[:,'charges']))



print('''
Age, low +ve skewness is less but -ve kurtosis is high in the distribution representing a flat plateu or distribution of data
Negative values of kurtosis indicate that a distribution is flat and has thin tails

BMI, low +ve  skewness and low negetive kurtosis. Very close to normal distribution

Charges, high positive skewness, long tail towards right, more outliers towards right.
high positive values of kurtosis indicate that a distribution is peaked and possess thick tails
''')

### g. Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

plt.figure(figsize= (10,8))
plt.subplot(3,1,1)
sns.boxplot(x=df.age)
plt.subplot(3,1,2)
sns.boxplot(x=df.bmi)
plt.subplot(3,1,3)
sns.boxplot(x=df.charges)

plt.show()


Box plots shows 
- Age - No outliers
- BMI - Few outliers for higher bmi values
- Charges - High outliers count towards high charges range


#getting 2 sigma values
print("Age:\n",df[(df.age>df.age.mean()+2*df.age.std())|(df.age<df.age.mean()-2*df.age.std())])
print("\nBMI:\n")
#df[(df.bmi>df.bmi.mean()+2*df.bmi.std())|(df.bmi<df.bmi.mean()-2*df.bmi.std())]
print(df[(df.bmi>df.bmi.mean()+3*df.bmi.std())|(df.bmi<df.bmi.mean()-3*df.bmi.std())])
print("\nCharges:\n")
print(df[(df.charges>df.charges.mean()+3*df.charges.std())|(df.charges<df.charges.mean()-3*df.charges.std())])



#### Using 2/3 sigma as well we can notice there are outlier present for BMI and Charges but, No outliers present in Age for even 2 sigma values


### 3h. Distribution of categorical columns (include children)

plt.figure(figsize= (12,12))
plt.subplot(2,2,1)
sns.countplot(x=df.sex)

plt.subplot(2,2,2)
sns.countplot(x=df.children)

plt.subplot(2,2,3)
sns.countplot(x=df.smoker)

plt.subplot(2,2,4)
sns.countplot(x=df.region)

plt.show()


From over all view, 
- Sex is distributed almost evenly,
- presence of no children/ less than 3 children is more compared to 3/4/5
- Smoker presene is less compared non-smokers
- Values are evenly spread across almost all regions


#### Analysis of  categorical column

sns.countplot(x=df.sex)
plt.show()
print(df.sex.value_counts())
print('''
Presence of male and female distribution looks very close similar, where male having slightly higher count then female
Based on the value counts, males are just 14 count above female distribution
''')



sns.countplot(x=df.children)
plt.show()
print(df.children.value_counts())
print('''
Presence of people having no children is higest compared to any other values.
Having 4/5 children is very less in distribution.
''')

sns.countplot(x=df.smoker,hue=df.sex)
plt.show()
print("Presence of non-smoker are high in the distribution, male smokers are high compared to female")
print(df.smoker.value_counts())

sns.countplot(x=df.smoker,hue=df.region)
plt.show()
print('''Smokers are distributed in all regions with southwest region having more somkers compared to others.
 Southwest is also leading the non smoker group as well with very light margin''')

sns.countplot(x=df.region)
plt.show()
print(df.region.value_counts())
print('''
Southwest region has slightly more dominance in distribution compared to other regions.
''')



### 3i. Pair plot that includes all the columns of the data frame

#pair plot ignores string values

# Deep copy
copydf=copy.deepcopy(df)
#covert categorical values to numerical before pair plot
copydf.loc[:,['sex', 'smoker', 'region']] = copydf.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 
plt.show()


sns.pairplot(copydf, kind="reg")
plt.show()

print('''
Overview:
Pattern with age and charges: formation of groups based on charges across all age groups
increase with age shows correlation with charges
Pattern with smoker and charges: charges are higher for smokers, correlations seems high
''')

### Detailed analysis

sns.pairplot(df,hue="smoker")
plt.show()
print('''
Non smokers presence is more in less than 10K charges and smokers has high presence in high chareges
Presence of smokers is less in people having 4/5 childrens
BMI peak at mean is slightly higher in non smoker compared to smokers
''')

sns.pairplot(df,hue="sex")
plt.show()
print('''
Male female distribution is similar in age,bmi,children and chareges.
Females are having slighlty higher presence in low charges (less than 25K) range compared to males.
''')

sns.pairplot(df,hue="region")
plt.show()
print('''
Northwest and southwest region presnce is more in less thank 25K charges range compared to other regions.
BMI peaks towards low BMI of northwest and southwest is higher compared to other regions, 
southeast is having lowest bmi peak of all and towards the high bmi region.
''')



#grid = sns.PairGrid(df)
#
#grid = grid.map_diag(plt.hist)
#grid = grid.map_upper(plt.scatter)
#grid = grid.map_lower(sns.kdeplot)





### 4a. Do charges of people who smoke differ significantly from the people who don't?

sns.swarmplot(x=df.smoker,y=df.charges)
plt.show()
print("Charges seems higher for smokers in initial view.")

# H0 - Charges of people are similar for smoker and non smoker.
# HA - Charges of people who smoke differ significantly from the people.

smoker_charges = df[df['smoker'] == 'yes'].charges
smoker_charges

non_smoker_charges = df[df['smoker'] == 'no'].charges
non_smoker_charges

print("Smoker charges:",smoker_charges.mean(),"\nNon Smoker Charges:",non_smoker_charges.mean())
print("Smoker average charge seems higher")

df.smoker.value_counts()

# Ho = "Charges of people are similar for smoker and non-smoker"
# Ha = "Charges of peple are different for smoker and non-smoker"
# considering alpha of 0.05 , significance level of 5%



t_sm,p_value_sm=stats.ttest_ind(smoker_charges,non_smoker_charges)

if(p_value_sm<0.05):
    print('''p-Values is less than 5% significane level or 0.05,
        i.e. we reject the null hypothesis of charges being same for smoker and non-smoker.
        Ha = "Charges of people are different for smoker and non-smoker"
        \np-values: ''',p_value_sm,
        '''\nt-Stats:''',t_sm)
else:
    print('''p-Values is greater than 5% significane level or 0.05, i.e. we fail to reject the null hypothesis of charges being same for smoker and non-smoker.
        \np-values: ''',p_value_sm,
        '''\nt-Stats:''',t_sm)
    

print('''Yes, charges of people who smoke differ significantly from the people who don't.
Statistically proven as p-value captured is significantly below 5% significance level.''')
print("Positive t stat show charges of smoker are higher then non-smoker.")



### 4b. Does bmi of males differ significantly from that of females?

sns.swarmplot(y=df.bmi,x=df.sex)
plt.show()
print("BMI distribution looks similar for both genders")

sns.scatterplot(x=df.bmi, y=df.age, hue=df.sex)

# Ho = "BMI is similar for males and females"
# Ha = "BMI is not similar for males and females"
# considering alpha of 0.05 , significance level of 5%

male_bmi=df[df['sex']=='male'].bmi
male_bmi.mean()

female_bmi=df[df['sex']=='female'].bmi
female_bmi.mean()

# mean BMI looks close enough for male and femal

t_bmi,p_value_bmi=stats.ttest_ind(female_bmi,male_bmi)
print(p_value_bmi)
print(t_bmi)

if(p_value_bmi<0.05):
    print('''p-Values is less than 5% significane level or 0.05, 
    i.e. we reject the null hypothesis of BMI is similar for males and females.
    Ha = "BMI is not similar for males and females"
    \np-values: ''',p_value_bmi,
    '''\nt-Stats:''',t_bmi)
else:
    print('''p-Values is greater than 5% significane level or 0.05, 
    \n i.e. we fail to reject the null hypothesis of BMI is similar for males and females.
    Ho = "BMI is similar for males and females"
    \np-values: ''',p_value_bmi,
    '''\nt-Stats:''',t_bmi)





### 4.c Is the proportion of smokers significantly different in different genders?

sns.countplot(x=df.smoker, hue=df.sex)

print("There is slight count difference between male and female smokers.")

#Chi-squared test for nominal (categorical) data

crosstab = pd.crosstab(df['sex'],df['smoker'])  # Contingency table of sex and smoker attributes

chi, p_value_sesm, dof, expected =  stats.chi2_contingency(crosstab)
print(p_value_sesm)

#Ho = Proportion of smokers are similar in different genders
#Ha = Proportion of smokers significantly different in different genders
# considering alpha of 0.05 , significance level of 5%

if p_value_sesm < 0.05:  # Setting our significance level at 5%
    print("p-Values is less than 5% significane level or 0.05, Reject the null hypothesis")
    print("Ha = Proportion of smokers significantly different in different genders")
else:
    print("p-Values is greater than 5% significane level or 0.05, Failed to reject null hypothesis")
    print("Ho = Proportion of smokers are similar in different genders")





### 4d. Is the distribution of bmi across women with no children, one child and two children, the same?

df_female = copy.deepcopy(df[df['sex'] == 'female'])
df_female

print(df_female[df_female['children'] == 0]['bmi'].mean())
print(df_female[df_female['children'] == 1]['bmi'].mean())
print(df_female[df_female['children'] == 2]['bmi'].mean())

print("Mean BMI looks similar for all 3 ")

zero = df_female[df_female.children == 0]['bmi']
one = df_female[df_female.children == 1]['bmi']
two = df_female[df_female.children == 2]['bmi']

# Ho = distribution of bmi across women with no children, one child and two children, the same
# Ha = distribution of bmi across women with no children, one child and two children, not the same
# considering alpha of 0.05 , significance level of 5%


f_stat, p_value_fm = stats.f_oneway(zero,one,two)

if p_value_fm < 0.05:
    print("p-Values is less than 5% significane level or 0.05, Reject the null hypothesis.")
    print(p_value_fm)
    print("Ha = distribution of bmi across women with no children, one child and two children, is not same.")
else:
    print("p-Values is greater than 5% significane level or 0.05, Failed to reject null hypothesis.")
    print(p_value_fm)
    print("Ho = distribution of bmi across women with no children, one child and two children, is same.")



