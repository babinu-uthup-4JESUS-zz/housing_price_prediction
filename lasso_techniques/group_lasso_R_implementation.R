getwd()
setwd("/Users/babs4JESUS/Documents/GitHub/housing_price_prediction/data_analysis")


X_data = read.csv('x_data_r.csv')
Y_data = read.csv('y_data_r.csv')

# Pass on the indices to the program. Make sure that all predictors corresponding to
# a categorical variable has the same index value.
ind = c(NA, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37,  38,  38,  38,  41,  42,  42,  44,  44,  44,  47,  47,  47,  50,
                      51,  51,  51,  51,  51,  51,  51,  51,  51,  51,  51,  51,  51,
                      51,  51,  51,  51,  51,  51,  70,  70,  70,  73,  74,  74,  74,
                      74,  78,  78,  78,  78,  82,  82,  84,  85,  85,  85,  85,  85,
                      85,  85,  85,  85,  94,  94,  94,  94,  94,  94,  94,  94,  94,
                      103, 103, 103, 106, 106, 108, 108, 108, 111, 111, 111, 114, 114,
                      114, 117, 117, 119, 119, 119, 122, 122, 122, 122, 122, 127, 127,
                      127, 127, 131, 132, 132, 132, 135, 136, 136, 138, 138, 138, 141,
                      141, 141, 144, 144, 144, 147, 147, 149, 149, 151, 151, 153, 153,
                      155, 155, 157, 157, 157)

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
