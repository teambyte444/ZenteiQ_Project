# A PROGRESS REPORT
## ZENTEIQ AI-ML INTERNSHIP

## Links are here, Check from here:
- [Project Report] (https://drive.google.com/file/d/19MloXFDRTzW2QLOIB5Dc2mSMLVp4Nc3-/view?usp=sharing)
- [![Colab]](https://colab.research.google.com/drive/1ck4WTvc7tYymQtP1qXW1wU-imCARjPl-?usp=sharing)

#

# Problem Statement: No. 6
Develop a machine learning model that can predict student dropout rates or academic success based on a variety of factors, such as attendance, grades, and demographic data. The model should be able to identify students who are at risk of dropping out or falling behind and provide targeted interventions and support.

# Team Name: 
      #�TEAM-BYTE�

# Team Members:
	SUBHAJIT DAS
	SUMIT HALDER
	SOUMEN KAYAL
	MEGHANA M.


# Data Collection and Importing Necessary Modules and Libraries:

As we don�t have access to any curated, private data that�s why we have collected a dataset from kaggle and then we worked on that dataset.
Then we have imported necessary libraries to work with data, visualize data, and train machine learning model.
#
Numpy (alias np) is a library for working with arrays and matrices of numerical data.
Pandas (alias pd) is a library for data manipulation and analysis.
Matplotlib.pyplot (alias plt) is a library for data visualization.
Plotly.graph_objects is a library for creating interactive plots.
Plotly.express is a high-level interface for creating visualizations.
Seaborn is a library for data visualization that is built on top of matplotlib.
LabelEncoder is a class for encoding categorical data.
Train_test_split is a function to split data into training and testing sets.
Accuracy_score is a function to evaluate the accuracy of a machine learning model.
Cross_val_score is a function to evaluate the performance of a machine learning model using cross-validation.
Pickle is a module for serializing and de-serializing Python objects.
Warnings is a module to handle warnings in Python.

Then using read_csv() we have loaded the CSV file into the DataFrame, so that we can perform various operations such as data cleaning, data manipulation, data analysis, and data visualization on the data.

Then using info() method, we got a glimpse of the details of the overall dataset which includes 

* The column name
* The number of non-null values in the column
* The data type of the column
* The amount of memory used by the column

# Exploratory Data Analysis:
Then we have done EDA to visualize the dataset. Bar chart and pie chart has been used to visualize the correlation between different columns. We have displayed marital status, economic status and other features on chart to get a better understand of the overall dataset. We have graphs and charts below:

# Model Training and Result Analysis:

By using LabelCoder(), we have converted the text or char data into numerical values.
We have used StandardScaler class from the preprocessing module of the sklearn (Scikit-learn) library, then selects four columns from a pandas DataFrame student_data and assigns them to a variable X. These four columns are 'Unemployment rate', 'Inflation rate', 'GDP', and 'Age at enrollment'. This suggests that the data in these columns will be preprocessed using the StandardScaler class. 
The StandardScaler class standardizes the data by subtracting the mean from each feature and then dividing by its standard deviation. This process results in a new feature distribution with a mean of zero and a standard deviation of one. This is useful for algorithms that assume that the input data has a Gaussian (normal) distribution, such as linear regression, logistic regression, and SVM.
A StandardScaler instance called scaler is then created using scaler = StandardScaler().

Finally, the fit_transform() method of the scaler object is called on the feature matrix X to standardize the data. The scaled data is then assigned to a variable scaled. The fit_transform() method fits the scaler to the data and then transforms the data using the scaler's parameters. The resulting scaled variable contains the transformed data that is ready to be used for machine learning tasks.

We have split the data in the ratio, Training Data : Testing Data = 8:2
We use the LogisticRegression class to create an instance of a logistic regression model with the name Softmax_reg.

multi_class is set to 'multinomial', indicating that the model is a multi-class classifier that uses the softmax function to compute probabilities for each class.

solver is set to 'lbfgs', which is an optimization algorithm used by the model to minimize the cost function during training.

C is set to 10, which is the inverse regularization strength. Regularization helps prevent overfitting by adding a penalty to the cost function, reducing the weights of the model.

After creating the instance, the fit() method is called on the model, passing in X_train and y_train as the training data for the model to learn from. The model is trained to minimize the cost function using the specified solver and regularization strength.

Overall, this code creates and trains a logistic regression model that can classify multiple classes using the softmax function with regularization.
# Till now, we got 91.32% accuracy(dropping 'Enrolled') and 76.04% accuracy (without dropping 'Enrolled'), and we are also working on it for expecting more accuracy.

# Contributions of the Team Members:

Sumit Halder: Data Collection and Pre-processing.
Subhajit Das: Explotary Data analysis and label encoding.
Soumen Kayal: Scaling and model training.
Meghana M.: Web-applicaton development on project and Classification selecton.

Moreover, all of us worked as a whole team. We also helped each other all the time. It's slightly difficult to differentiate each other's�contributions.

# Conclusion:

We are working on it and putting heart and soul effort to enhance the accuracy of the model. We are also trying to integrate the model with a web�application.
