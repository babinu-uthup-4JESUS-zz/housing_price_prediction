import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn import linear_model

def add(a, b):
    return (np.abs(a) + b)


def evaluate_model_score(my_model, X, Y):
    predictions = my_model.predict(X)
    return evaluate_model_score_given_predictions(predictions, Y)

def evaluate_model_score_given_predictions(predictions, Y):
    mean_of_squared_error1 = \
        mean_squared_error(np.log(np.abs(Y)), np.log(np.abs(predictions)))
    return np.sqrt(mean_of_squared_error1)

def evaluate_neg_model_score(my_model, X, Y):
    return (-1) * evaluate_model_score(my_model, X, Y)

def cross_val_score_given_model(my_model, X, Y, cv=5):
    cross_val_score1 = cross_val_score(my_model, 
                                       X, Y, 
                                       scoring=evaluate_model_score, 
				       cv=cv)
    return cross_val_score1.mean()

def fit_pipeline_and_cross_validate(my_pipeline,
				    train_data, 
		   		    X_columns, 
				    Y_column='SalePrice'):
    X = train_data[X_columns]
    Y = train_data[[Y_column]]
    my_pipeline.fit(X, Y)


    return (my_pipeline, cross_val_score_given_model(my_pipeline, X, Y))

def print_model_stats_from_pipeline(pipeline_obj, 
				    cross_validation_score, 
				    print_coef=False, 
				    print_intercept=True):
    print("Cross validation score is {0}".format(cross_validation_score))
    clf_model = pipeline_obj.named_steps['model']

    if print_intercept:
        print("Intercept of the model is {0}".format(clf_model.intercept_))  
    if print_coef:
        print("Lasso model coefficients are {0}".format(clf_model.coef_))    
    print("Number of predictors in the model is {0}".format(np.sum(clf_model.coef_ != 0)))    
    


