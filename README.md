# Group-Loans-Lending-Using-Machine-Learning
## About The Project
Diversification is known to be important to lending, but not much is known about their relative performance in giving loans and whether they’d default or not, especially when they are in conflict. Using the group loan structure in India, we find that in an economic setting dominated by information asymmetry, heterogeneity in groups leads to better performances in paying the Loan  when compared to homogeneity. Using machine learning techniques like XGBoost and Random Forest, the paper shows that social ties increases the risk of default.
## Task/Goals
-We were supposed to develop a tool using Machine Learning techniques that will tell us(the banks) whether the group will default or not by using features like caste homogeneity, occupation homogeneity, education, expenditure, income etc.

-We were supposed to do regression plot between the actual defaults in total 25 different collection(bins) of 30 groups and the probability of default of those collections(bins)
## Challenges We Faced
Data was provided but the major problem we faced with it was cleaning and selecting the features with more relevance and the features which has less multicollinearity problems. We did it in order to avoid severe multicollinearity because severe multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model 
## Implementation
-For the task related to default prediction tool, we first did feature engineering which included formation of features like educational score, dummy variables and we did padding(incorporating the overall effect of a categorical variable with many categories in specified number of variables) of some of the variables. We also curtailed the amount of 0(not a default cases) into the train_test_split so our model learns the best possible fit

-After performing this, we trained the data using different models 

      -First we went with XGBoost classifier and trained it on the best parameters which was searched by Randomized search CV
      
      -We also tried Artificial Neural Networks with two layers and iterated it over the different combinations of number of nodes in the two layers to get the best result
      
      -We finally concluded that XGBoost classifier works best in this type of data as it avoids overfitting and multicollinearity upto some extent because of bagging and bi-node formation  
      
-Now, for the task related to regression plot. We found the probability of default of the groups in the test set and then divided it into groups of 25 loan groups each and then calculated the total number of defaults in each of those groups of 25 loan groups and then we made a regression plot with both the mean probability  and total defaults for each of the group of 25 loan groups. 
## Results
Best Results  were obtained from XGBoost model with a set of best parameters which was given by Randomised Search CV

-ROC_AUC_score: 72.6 to 74.9%

-Fscore: 57.4 to 60.8%

-Precision: 49.9 to 52.5%

-Recall: 55.4 to 57.6%

-Gini: 42.6 to 45.7%
