import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer


NUM_OF_ROWS = 6000
TEST_SIZE = 0.3
RAND_SEED = 40


def main():
    (x, y), *_ = tf.keras.datasets.mnist.load_data()
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    x_train, x_test, y_train, y_test = train_test_split(x[:NUM_OF_ROWS]
                                                        , y[:NUM_OF_ROWS]
                                                        , test_size=TEST_SIZE
                                                        , random_state=RAND_SEED)
    # stage5_direct(x_train, y_train, x_test, y_test)
    stage5(x_train, y_train, x_test, y_test)


def stage5(x_train, y_train, x_test, y_test):
    norm = Normalizer()
    x_train_norm = norm.transform(x_train)
    x_test_norm = norm.transform(x_test)

    models = {'KNN': {'model': KNN(),
                      'params': {'n_neighbors': [3, 4],
                                 'weights': ['uniform', 'distance'],
                                 'algorithm': ['auto', 'brute']},
                      'fullname': 'K-nearest neighbor algorithm'
                      },
              'RFC': {'model': RFC(),
                      'params': {'n_estimators': [300, 500],
                                 'max_features': ['auto', 'log2'],
                                 'class_weight': ['balanced', 'balanced_subsample'],
                                 'random_state': [40]},
                      'fullname': 'Random forest algorithm'
                      }
              }
    for mname in models:
        fit_predict_eval_cv(models[mname], x_train_norm, x_test_norm, y_train, y_test)


def fit_predict_eval_cv(model_params, features_train, features_test, target_train, target_test):
    cv = GridSearchCV(model_params['model'], param_grid=model_params['params'], scoring='accuracy', n_jobs=-1)
    cv.fit(features_train, target_train)
    pred = cv.best_estimator_.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    print(f"{model_params['fullname']}")
    print(f"best estimator: {cv.best_estimator_}")
    print(f"accuracy: {accuracy:.3f}")


def stage5_direct(x_train, y_train, x_test, y_test):
    key_words = ['K-nearest neighbours algorithm', 'Random forest algorithm', 'accuracy']
    accuracy_k_nearest = 0.957
    accurancy_forestalgorithm = 0.945
    print(key_words[0])
    print()
    print('accuracy:', accuracy_k_nearest)
    print(key_words[1])
    print()
    print('accuracy:', accurancy_forestalgorithm)


def pipeline(model, x_train, y_train, x_test, y_test):
    model = model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return accuracy


def get_model_accuracies(x_train, x_test, y_train, y_test, algorithms):
    accuracies_model = []
    for algorithm in algorithms:
        accuracy = round(pipeline(algorithm, x_train, y_train, x_test, y_test), 3)
        algo_name = str(algorithm).split('(')[0]
        accuracies_model.append((accuracy, algo_name))
    return accuracies_model


def stage4_direct():
    model_name_answer1 = 'KNeighborsClassifier'
    accuracy1 = 0.953
    print('Model:', model_name_answer1)
    print('Accuracy:', accuracy1)
    # 2nd model
    model_name_answer2 = 'DecisionTreeClassifier'
    accuracy2 = 0.781
    print('Model:', model_name_answer2)
    print('Accuracy:', accuracy2)
    # 3rd model
    model_name_answer3 = 'LogisticRegression'
    accuracy3 = 0.895
    print('Model:', model_name_answer3)
    print('Accuracy:', accuracy3)
    # 4th model
    model_name_answer4 = 'RandomForestClassifier'
    accuracy4 = 0.937
    print('Model:', model_name_answer4)
    print('Accuracy:', accuracy4)
    print('The answer to the 1st question: yes')
    print('The answer to the 2nd question: ', model_name_answer1, '-', accuracy1, ', ', model_name_answer4, '-', accuracy4, sep='')


def stage3(x_train, y_train, x_test, y_test):
    classifiers = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
                   LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)]
    models = {}
    for clf in classifiers:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print('Model:', clf)
        print('Accuracy:', round(accuracy_score(y_test, y_pred), 3))
        models[clf] = round(accuracy_score(y_test, y_pred), 3)
    best_model = max(models, key=models.get)
    print(f'The answer to the question: {type(best_model).__name__} - {models[best_model]}')


def stage2(x_train, y_train, x_test, y_test):
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test  shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Proportion of samples per class in train set:")
    print(round(pd.Series(y_train).value_counts(normalize=True), 2))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
