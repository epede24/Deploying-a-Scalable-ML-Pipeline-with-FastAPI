# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a logistic regression model. The max number of iterations is set to 300. A grid search was used on a hyperparameter grid to select the best performing model for the use case. 
## Intended Use
The model evaluates the impact of categorical features on an individual's salary. The model should be used to predict if someone earns over or under 50k per year. 
## Training Data
The model was trained on census income data from 1994, obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). 

The dataset has 32561 rows and 15 columns. The dataset was split for training and testing, with 70% of the dataset was used for training. 

## Evaluation Data
The same dataset was used for testing. 30% of the dataset was reserved for evaluation. 

## Metrics
The model was evaluated using precision, recall, and the F1 score. 

The metrics are as follows: 
- Precision: 0.6708
- Recall: 0.5208 
- F1: 0.5863
## Ethical Considerations
An ethical consideration that must be made when working with census data is that the census overrepresents some groups while underrepresenting others. The categories of the census may not account for the way that individuals wish to describe their racial and ethnic identities, which can lead to further bias in the data. Census data, therefore, must not be considered to be a complete picture of a population's demographics. 
## Caveats and Recommendations
The census data used is from 1994, and therefore it is recommended that future efforts are made with updated data. Additionally, considering the bias inherent in census data, it is recommended that predictions made by the model are not thought of as wholly conclusive. 