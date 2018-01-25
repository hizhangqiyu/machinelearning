import tensorflow as tf
import pandas as pd
import itertools

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'medv']
FEATURES = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio']
LABEL = 'medv'

training_set = pd.read_csv('../../../dataset/boston/boston_train.csv', skipintitalspace=True, skprows=1, names=COLUMNS)
test_set = pd.read_csv('../../../dataset/boston/boston_test.csv', skipintitalspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv('../../../dataset/boston/boston_predict.csv', skipintitalspace=True, skiprows=1, names=COLUMNS)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[10,10], model_dir="../../../dataset/tensorboard")

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

regressor.train(input_fn=get_input_fn(training_set), steps=5000)

ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
loss_score = ev["loss"]
print('Loss: {0:f}'.format(loss_score))

y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print('Predictions: {}'.fromat(str(predictions)))



