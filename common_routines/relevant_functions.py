import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import matplotlib.pyplot as plt
FIGURE_LENGTH = 16
FIGURE_BREADTH = 9

def add(a, b):
    return (np.abs(a) + b)


def evaluate_model_score(my_model, X, Y):
    predictions = my_model.predict(X)
    return evaluate_model_score_given_predictions(predictions, Y)

def make_predictions(my_model, X):
    predictions = my_model.predict(X)
    return predictions

def evaluate_model_score_given_predictions(predictions, Y):
    mean_of_squared_error1 = \
        mean_squared_error((np.abs(Y)), (np.abs(predictions)))
    return np.sqrt(mean_of_squared_error1)

def evaluate_neg_model_score(my_model, X, Y):
    return (-1) * evaluate_model_score(my_model, X, Y)

def cross_val_score_given_model(my_model, X, Y, cv=5):
    cross_val_score1 = cross_val_score(my_model, 
                                       X, Y, 
                                       scoring=evaluate_model_score, 
				       cv=cv)
    return cross_val_score1.mean()

def cross_val_scores_given_model(my_model, X, Y, cv=5):
    cross_val_score1 = cross_val_score(my_model, 
                                       X, Y, 
                                       scoring=evaluate_model_score, 
				       cv=cv)
    return cross_val_score1

def fit_pipeline_and_cross_validate(my_pipeline,
				    train_data, 
		   		    X_columns, 
				    Y_column='LogSalePrice'):
    X = train_data[X_columns]
    Y = train_data[[Y_column]].values.ravel()
    my_pipeline.fit(X, Y)
    return (my_pipeline, cross_val_score_given_model(my_pipeline, X, Y))

def fit_pipeline_and_return_cross_validation_scores(
    my_pipeline,
	train_data, 
	X_columns, 
	Y_column='LogSalePrice'):
    X = train_data[X_columns]
    Y = train_data[[Y_column]].values.ravel()
    my_pipeline.fit(X, Y)
    return (my_pipeline, cross_val_scores_given_model(my_pipeline, X, Y))


def fit_pipeline_and_evaluate_on_validation_set(my_pipeline,
				    train_data, 
                    validation_data,
		   		    X_columns, 
				    Y_column='LogSalePrice'):
    X = train_data[X_columns]
    Y = train_data[[Y_column]].values.ravel()
    my_pipeline.fit(X, Y)
    
    X_validation = validation_data[X_columns]
    Y_validation = validation_data[[Y_column]].values.ravel()
    
    return (my_pipeline, evaluate_model_score(my_pipeline, X_validation, Y_validation))


def fit_pipeline_and_make_predictions_on_test_set(my_pipeline,
				    train_data, 
                    test_data,
		   		    X_columns, 
				    Y_column='LogSalePrice'):
    X = train_data[X_columns]
    Y = train_data[[Y_column]].values.ravel()
    my_pipeline.fit(X, Y)
    
    X_test = test_data[X_columns]
    
    return (my_pipeline, make_predictions(my_pipeline, X_test))

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

# Goes over the input date and drop columns which have too many null(around 20%
# ). Also checks if some data transformations could help here.
def get_validated_transformed_data(input_csv_file):
    complete_train_data = pd.read_csv(input_csv_file)
    null_vals_per_col = list(complete_train_data.isnull().values.sum(axis=0))
    col_names = list(complete_train_data.columns)
    cols_to_num_null_vals = dict(zip(col_names, null_vals_per_col))
    cols_with_many_null_entries = \
       [col for col in cols_to_num_null_vals.keys() if cols_to_num_null_vals.get(col) > 250]
        
    # Also, make sure that we drop column Id as well, since it does not give us any predictive value.
    print(cols_with_many_null_entries)
    cols_with_many_null_entries.append('Id')
    complete_train_data.drop(cols_with_many_null_entries, inplace=True, axis=1)
    complete_train_data['LogLotArea'] = complete_train_data['LotArea'].apply(lambda x : np.log(x))
    return complete_train_data

def get_null_value_details(given_df):
    num_null_vals = given_df.isnull().values.sum()
    print("Total number of null values in training data is {0} ".format(num_null_vals))    
    null_vals_per_col = list(given_df.isnull().values.sum(axis=0))
    col_names = list(given_df.columns)
    cols_to_num_null_vals = dict(zip(col_names, null_vals_per_col))
    print("\nNULL VALUES FOR EACH COLUMN")
    for col,num_null_val in cols_to_num_null_vals.items():
        if num_null_val != 0:
            print(col, num_null_val)

def get_cross_validation_score(pipeline_obj, 
                               train_data_one_hot, 
                               predictor_cols):
    (my_pipe, cross_validation_score) = fit_pipeline_and_cross_validate(
        pipeline_obj, train_data_one_hot, predictor_cols)
    return cross_validation_score

def plot_relevant_df(predictor_index_cross_val_score_df, 
                    title='Cross Validation score vs Predictor column index',
                    fig_length=FIGURE_LENGTH,
                    fig_breadth=FIGURE_BREADTH):
    fig, ax = plt.subplots(1, 1, figsize=(fig_length, fig_breadth))
    predictor_index_cross_val_score_df.plot(ax=ax)
    ax.set_title(title)
    return ax

def get_predictor_df(index_name, 
                     compute_func,
                     index_start,
                     predictor_cols):
    NUM_POINTS = len(predictor_cols)
    predictor_index_cross_val_score_df = pd.DataFrame(np.arange(index_start,NUM_POINTS + index_start), columns=[index_name])
    predictor_index_cross_val_score_df['cross_val_score'] = \
        predictor_index_cross_val_score_df[index_name].apply(lambda x : compute_func(x))
    predictor_index_cross_val_score_df.index = predictor_index_cross_val_score_df[index_name]
    predictor_index_cross_val_score_df.drop(columns=[index_name], inplace=True)
    return predictor_index_cross_val_score_df
