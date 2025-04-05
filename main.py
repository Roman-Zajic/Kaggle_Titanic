import pandas as pd
import xgboost as xgb
#from pyexpat import features
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':

    model = xgb.XGBClassifier()

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    features = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']


    # preprocessing
    le_sex = LabelEncoder()
    df_train['Sex'] = le_sex.fit_transform(df_train['Sex'])
    df_test['Sex'] = le_sex.fit_transform(df_test['Sex'])

    x = df_train[features]
    y = df_train['Survived']

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x, y)

    # Best parameters and accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    best_model.fit(x, y)

    # predict
    df_test['Survived'] = best_model.predict(df_test[features])

    result = df_test[features + ['Survived']]
    submission = df_test[['PassengerId', 'Survived']]

    result.to_csv('result.csv')
    submission.to_csv('submission.csv', index=False)

    # evaluate
    y_pred = best_model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')




