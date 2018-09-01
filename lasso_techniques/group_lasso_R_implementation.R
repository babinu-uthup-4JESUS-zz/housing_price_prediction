getwd()
setwd("/Users/babs4JESUS/Documents/GitHub/housing_price_prediction/data_analysis")


X_data = read.csv('x_data_r.csv')
Y_data = read.csv('y_data_r.csv')

# Pass on the indices to the program. Make sure that all predictors corresponding to
# a categorical variable has the same index value.
ind = c(NA, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37,  37,  37,  40,  41,  41,  43,  43,  43,  46,  46,  46,  49,
        50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,
        50,  50,  50,  50,  50,  50,  69,  69,  69,  72,  73,  73,  73,
        73,  77,  77,  77,  77,  81,  81,  83,  84,  84,  84,  84,  84,
        84,  84,  84,  84,  93,  93,  93,  93,  93,  93,  93,  93,  93,
        102, 102, 102, 105, 105, 107, 107, 107, 110, 110, 110, 113, 113,
        113, 116, 116, 118, 118, 118, 121, 121, 121, 121, 121, 126, 126,
        126, 126, 130, 131, 131, 131, 134, 135, 135, 137, 137, 137, 140,
        140, 140, 143, 143, 143, 146, 146, 148, 148, 150, 150, 152, 152,
        154, 154, 156, 156, 156)

# Convert to matrices as the routine works only with them.
library(grplasso)
X_data_matrix  = as.matrix(X_data)
Y_data_matrix  = as.matrix(Y_data)

# Find the maximum possible value of lambda and create a sequence of values using the same.
maxlam = lambdamax(X_data_matrix, Y_data_matrix, ind, model=LinReg())
NUM_SEQ_VALUES = 25
lamseq = maxlam * (0.75 ^(0:NUM_SEQ_VALUES))

# Create lasso models for different values of lambda and pick the best amongst them.
holdout = grplasso(X_data_matrix, Y_data_matrix, ind, lambda = lamseq, model = LinReg(), standardize = TRUE)
holdout$coefficients
holdout
apply(holdout$coefficients!=0,2,sum)*2 + holdout$nloglik
length(lamseq)
lamseq[22]
final_model = grplasso(X_data_matrix, Y_data_matrix, ind, lambda = lamseq[15], model = LinReg())

# Create test data and make predictions on the same.
X_test_data = read.csv('x_test_data_r.csv')
predictions_test_data = predict(final_model, X_test_data)
write(predictions_test_data, 'predictions_test_data.csv')
  