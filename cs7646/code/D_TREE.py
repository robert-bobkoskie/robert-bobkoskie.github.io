# ASSIGNMENT 1
# Robert Bobkoskie
# rbobkoskie3


import os, re
import csv
import pydotplus
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import tree
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_iris
from sklearn.datasets import make_hastie_10_2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier


class Utilities:

    def Preprocess_Data(self, name, target, data_file, normalize_data):
        # PANDAS, read data, convert to csv
        in_file = open(data_file, 'r')
        out_file = open('DATA.csv', 'w')
        for line in in_file:
            line = line.replace(';', ',')
            out_file.write(line)
        out_file.close()

        df = pd.read_csv('DATA.csv', index_col=False)
        #print '* Class Labels (targets):', df[target].unique()

        if data_file == 'bank-additional.csv':
            # Clean data by deleting columns
            df.drop(['default', 'duration'], axis=1, inplace=True)
            # Create class names before mapping class names from str to int
            # Get the class labels from the target col
            classes = df[target].unique()

            # map strings to ints
            mapping = {'yes': 1,'no': 0,

                       'admin.': 0,'blue-collar': 1,'entrepreneur': 2,'housemaid': 3,'management': 4,'retired': 5,
                       'self-employed': 6,'services': 7,'student': 8,'technician': 9,'unemployed': 10,'unknown': 11,

                       'divorced': 0,'married': 1,'single': 2,
                       'mon': 0,'tue': 1,'wed': 2,'thu': 3,'fri': 4,
                       'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
                       'jul': 6,'aug': 7,'sep': 8,'oct': 9,'nov': 10, 'dec': 11,

                       'basic.4y': 0,'basic.6y': 1,'basic.9y': 2,'high.school': 3,'illiterate': 4,
                       'professional.course': 5,'university.degree': 6,

                       'cellular': 0,'telephone': 1,
                       'success': 1,'failure': 0,'nonexistent': 3,
                       'unknown': None}
            df.replace(mapping, inplace=True)
            df.dropna(axis=0, inplace=True)  #Del null data: axis=0 is row, axis=1 is col
            df = df.rename(columns={'emp.var.rate': 'emp_var_rate', 'cons.price.idx': 'cons_price_idx',
                                    'cons.conf.idx': 'cons_conf_idx', 'nr.employed': 'nr_employed'})

            features = list(df)
            features.remove(target) #Remove the target name from the list of features

            #TEST = np.meshgrid(np.arange(5), np.arange(5))[0]
            #print TEST
            #print '\nSLICE', TEST[:,:-1]
            #print '\nSLICE', TEST[:,-1:]
            mat = df.as_matrix()

            # print np.shape(mat[:,-1:])   # shape, (3811L, 1L) defines col as '1L'
            # reshape 1-d array using -1, shape will be (3811L,)
            #target = np.reshape(mat[:,-1:], -1)

            # All columns excpets the last col will be data
            data = mat[:,:-1]
            # Get the target column
            target = df[target]

            featr_names = features
            class_names = classes
            X = data
            y = target

            print '\n========================='
            print name
            print 'Data  Size: %d x %d' % (X.shape[0], X.shape[1])
            print '# Features: %d' % (len(featr_names))
            print '#  Classes: %d' % (len(class_names))
            print '========================='

            #######################
            # Misc. pandas dataframe verification
            #######################
            #print type(mat), mat.shape
            #print df.head(n=5)
            #print df.tail(n=5)
            #print df.columns
            #print df
            #print df['housing']
            #print df.dtypes
            #print df.shape

        elif data_file == 'spamdata.csv':
            classes = df.iloc[:,-1].unique()
            features = list(df)
            mat = df.as_matrix()
            data = mat[:,:-1]
            target = df.iloc[:,-1]
            featr_names = features
            class_names = classes
            class_names = class_names.astype('|S2')
            #print type(featr_names), type(featr_names[0]), type(class_names), type(class_names[0])
            X = data
            y = target

        elif data_file == 'OR.csv':
            classes = df.iloc[:,-1].unique()
            features = list(df)
            mat = df.as_matrix()
            data = mat[:,:-1]
            target = df.iloc[:,-1]
            featr_names = features
            class_names = classes
            class_names = class_names.astype('|S2')
            #print type(featr_names), type(featr_names[0]), type(class_names), type(class_names[0])
            X = data
            y = target

        # Save Data File
        if normalize_data:   #MIN-MAX Scaling
            result = df.copy()
            for feature_name in df.columns:
                #print feature_name
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            #print result
            result.to_csv(name+".txt", index=False, header=False)
            print name
            print 'In \'RandOpt.java\', num_instances = ', result.shape[0]
            print 'In \'RandOpt.java\', num_attributes = ', result.shape[1] - 1

        else:
            #print df.iloc[[0]]
            #df.drop(df.index[[0]], axis=0, inplace=True)
            df.to_csv(name+".txt", index=False, header=False)
            print name
            print 'In \'RandOpt.java\', num_instances = ', df.shape[0]
            print 'In \'RandOpt.java\', num_attributes = ', df.shape[1] - 1

        return X, y, featr_names, class_names


    def Plot_Decision_Tree(self, name, features, classes, classifier):
        tree.export_graphviz(classifier, out_file=str(name)+'_d_tree.dot',
                             feature_names=features,
                             class_names=classes,
                             filled=False, rounded=True,
                             impurity=True)

        graph = pydotplus.graph_from_dot_file(str(name)+"_d_tree.dot")
        graph.write_png(str(name)+"_d_tree.png")


    def Statistics(self, y_test, y_pred):
        print('Misclassified samples: %d' % (y_test != y_pred).sum())
        print('num Correct: %d' % accuracy_score(y_test, y_pred, normalize=False))
        print('pct Correct: %.2f\n' % accuracy_score(y_test, y_pred))


    def Grid_Search(self, clf, name, X_train, y_train, hyper_params, num_folds):
        out_file = open(name+'_PARAMS.txt', 'w')

        scores = ['recall']#, 'precision']
        for score in scores:
            print '# Tuning hyper-parameters for %s' % (score)

            grid_search = GridSearchCV(clf,
                                       param_grid=hyper_params,
                                       cv=num_folds,
                                       scoring='%s_macro' % score)

            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_

            print  'Best hyper-parameters found on training set:'
            print best_params, '\n'

            means = grid_search.cv_results_['mean_test_score']
            stds = grid_search.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
                line = ("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
                out_file.write(line)
                #print("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))

        out_file.close()
        return best_params


    def Graph_Boosted(self, name,
                      n_trees_discrete, n_trees_real,
                      bdt_discrete, bdt_real,
                      discrete_test_errors, real_test_errors,
                      discrete_estimator_errors, real_estimator_errors,
                      discrete_estimator_weights):

        #######################
        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py
        #######################
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        y_min = 0.75*min(np.amin(real_test_errors), np.amin(discrete_test_errors))
        y_max = 1.25*max(np.amax(real_test_errors), np.amax(discrete_test_errors))
        plt.plot(range(1, n_trees_discrete + 1),
                 discrete_test_errors, c='black', label='SAMME')
        plt.plot(range(1, n_trees_real + 1),
                 real_test_errors, c='black',
                 linestyle='dashed', label='SAMME.R')
        plt.legend()
        plt.ylim(y_min, y_max)
        plt.ylabel('Test Error')
        plt.xlabel('Number of Trees')

        plt.subplot(132)
        plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
                 "b", label='SAMME', alpha=.5)
        plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
                 "r", label='SAMME.R', alpha=.5)
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2,
                  max(real_estimator_errors.max(),
                      discrete_estimator_errors.max()) * 1.2))
        plt.xlim((-20, len(bdt_discrete) + 20))

        plt.subplot(133)
        plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
                 "b", label='SAMME')
        plt.legend()
        plt.ylabel('Weight')
        plt.xlabel('Number of Trees')
        plt.ylim((0, discrete_estimator_weights.max() * 1.2))
        plt.xlim((-20, n_trees_discrete + 20))

        # prevent overlapping y-axis labels
        plt.subplots_adjust(wspace=0.25)
        #plt.show()
        plt.savefig(str(name)+"_d_tree_boosted.png")
        plt.close()

    def plot_learning_curve(self, name, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        #plt.show()
        plt.savefig(str(name)+"_"+str(title)+".png")
        plt.close()


    def Standard_Scale(self, X_train, X_test):
        #######################
        # STANDARDIZE DATA
        #######################
        sc = StandardScaler()
        X_trn_std = sc.fit_transform(X_train)
        X_tst_std = sc.transform(X_test)

        return X_trn_std, X_tst_std


class SupervisedLearning:

    my_utils = Utilities()
    make_graph = True
    more_stats = True
    graph_lc = False   #Learning curves for SVM (with polyfit) crash the PC

    def Classify(self, X, y, X_train, X_test, y_train, y_test,
                 name, featr_names, class_names, best_params, options):

        #######################
        # Classify Data
        #######################
        if options['Classify'] == 'd_tree':
            print 'D Tree'
            d_tree = DecisionTreeClassifier(criterion=best_params['criterion'],
                                            max_depth=best_params['max_depth'],
                                            min_samples_split=best_params['min_samples_split'],
                                            min_samples_leaf=best_params['min_samples_leaf'],
                                            random_state=None,
                                            max_leaf_nodes=best_params['max_leaf_nodes'])

            cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            self.my_utils.plot_learning_curve(name, d_tree, 'Learning Curve (D Tree)', X, y,
                                              ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(d_tree, X_train, y_train)
                print('Cross Val Score Training Set (d_tree): %.2f' % scores.mean())
                scores = cross_val_score(d_tree, X_test, y_test)
                print('Cross Val Score Test Set (d_tree): %.2f' % scores.mean())

            d_tree.fit(X_test, y_test)

            #######################
            # GRAPHING and ACCURACY
            #######################
            if self.make_graph:
                self.my_utils.Plot_Decision_Tree(name, featr_names, class_names, d_tree)

            # ACCURACY
            print('D Tree Score: %f' % d_tree.fit(X_train, y_train).score(X_test, y_test))
            y_pred = d_tree.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

        if options['Classify'] == 'd_tree' and options['Bagging']:
            #######################
            # BaggingClassifier
            # http://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator
            # RandomForestClassifier
            # http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
            #######################
            print 'Random Forrest (Bagging) D Tree'
            clf = RandomForestClassifier(n_estimators=10,
                                         criterion=best_params['criterion'],
                                         max_depth=best_params['max_depth'],
                                         min_samples_split=best_params['min_samples_split'],
                                         min_samples_leaf=best_params['min_samples_leaf'],
                                         random_state=None,
                                         max_leaf_nodes=best_params['max_leaf_nodes'])

            #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            #self.my_utils.plot_learning_curve(name, clf, 'Learning Curve (Bagging)', X, y,
            #                                  ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(clf, X_train, y_train)
                print('Cross Val Score Training Set (Rand Forst): %.2f' % scores.mean())
                scores = cross_val_score(clf, X_test, y_test)
                print('Cross Val Score Test Set (Rand Forst: %.2f' % scores.mean())

            # ACCURACY
            print('D Tree Score (Rand Forst): %f' % clf.fit(X_train, y_train).score(X_test, y_test))
            y_pred = clf.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

            print 'Bagging D Tree'
            clf = BaggingClassifier(DecisionTreeClassifier(
                                           criterion=best_params['criterion'],
                                           max_depth=best_params['max_depth'],
                                           min_samples_split=best_params['min_samples_split'],
                                           min_samples_leaf=best_params['min_samples_leaf'],
                                           random_state=None,
                                           max_leaf_nodes=best_params['max_leaf_nodes']),
                                        max_samples=0.5, max_features=0.5)

            #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            #self.my_utils.plot_learning_curve(name, clf, 'Learning Curve (Rnd Frst)', X, y,
            #                                  ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(clf, X_train, y_train)
                print('Cross Val Score Training Set (bagging): %.2f' % scores.mean())
                scores = cross_val_score(clf, X_test, y_test)
                print('Cross Val Score Test Set (bagging): %.2f' % scores.mean())

            # ACCURACY
            print('D Tree Score (Bagging): %f' % clf.fit(X_train, y_train).score(X_test, y_test))
            y_pred = clf.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

        if options['Classify'] == 'd_tree' and options['Boosted']:
            #######################
            # AdaBoostClassifier
            # http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py
            # http://scikit-learn.org/stable/modules/ensemble.html#usage
            #######################
            print 'Add a Boost D Tree'
            clf = AdaBoostClassifier(DecisionTreeClassifier(
                                        criterion=best_params['criterion'],
                                        max_depth=best_params['max_depth'],
                                        min_samples_split=best_params['min_samples_split'],
                                        min_samples_leaf=best_params['min_samples_leaf'],
                                        random_state=None,
                                        max_leaf_nodes=best_params['max_leaf_nodes']),
                                     n_estimators=600,
                                     learning_rate=1)

            #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            #self.my_utils.plot_learning_curve(name, clf, 'Learning Curve (Adda Bst)', X, y,
            #                                  ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(clf, X_train, y_train)
                print('Cross Val Score Training Set (adda boost): %.2f' % scores.mean())
                scores = cross_val_score(clf, X_test, y_test)
                print('Cross Val Score Test Set (adda boost): %.2f' % scores.mean())

            # ACCURACY
            print('D Tree Score (Adda Boost): %f' % clf.fit(X_train, y_train).score(X_test, y_test))
            y_pred = clf.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

            #n_split = 3000   #Does not work on a small data set like the IRIS
            n_split = np.shape(X_train)[0]
            X_train, X_test = X[:n_split], X[n_split:]
            y_train, y_test = y[:n_split], y[n_split:]
            # a = np.arange(6); print np.shape(a), np.shape(a.reshape(np.shape(a)[0],1))

            bdt_real = AdaBoostClassifier(DecisionTreeClassifier(
                                             criterion=best_params['criterion'],
                                             max_depth=best_params['max_depth'],
                                             min_samples_split=best_params['min_samples_split'],
                                             min_samples_leaf=best_params['min_samples_leaf'],
                                             random_state=None,
                                             max_leaf_nodes=best_params['max_leaf_nodes']),
                                          n_estimators=600,
                                          learning_rate=1)

            bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(
                                                 criterion=best_params['criterion'],
                                                 max_depth=best_params['max_depth'],
                                                 min_samples_split=best_params['min_samples_split'],
                                                 min_samples_leaf=best_params['min_samples_leaf'],
                                                 random_state=None,
                                                 max_leaf_nodes=best_params['max_leaf_nodes']),
                                              n_estimators=600,
                                              learning_rate=1.5,
                                              algorithm="SAMME")

            bdt_real.fit(X_train, y_train)
            bdt_discrete.fit(X_train, y_train)

            real_test_errors = []
            discrete_test_errors = []

            for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(X_test),
                                                                 bdt_discrete.staged_predict(X_test)):

                #pass
                real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))
                discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))

            n_trees_discrete = len(bdt_discrete)
            n_trees_real = len(bdt_real)

            discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
            real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
            discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

            self.my_utils.Graph_Boosted(name,
                                        n_trees_discrete, n_trees_real,
                                        bdt_discrete, bdt_real,
                                        discrete_test_errors, real_test_errors,
                                        discrete_estimator_errors, real_estimator_errors,
                                        discrete_estimator_weights)

            
            # GradientBoostingClassifier
            # http://scikit-learn.org/stable/modules/ensemble.html#classification
            #######################
            print 'Gradient Boosted D Tree'
            X, y = make_hastie_10_2(random_state=0)
            X_train, X_test = X[:2000], X[2000:]
            y_train, y_test = y[:2000], y[2000:]

            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                             max_depth=best_params['max_depth'],
                                             min_samples_split=best_params['min_samples_split'],
                                             min_samples_leaf=best_params['min_samples_leaf'],
                                             max_leaf_nodes=best_params['max_leaf_nodes'],
                                             random_state=0).fit(X_train, y_train)

            #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            #self.my_utils.plot_learning_curve(name, clf, 'Learning Curve (Grad Bst)', X, y,
            #                                  ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(clf, X_train, y_train)
                print('Cross Val Score Training Set (grad boost): %.2f' % scores.mean())
                scores = cross_val_score(clf, X_test, y_test)
                print('Cross Val Score Test Set (grad boost): %.2f' % scores.mean())

            # ACCURACY
            print('D Tree Score (Grad Boost): %f' % clf.fit(X_train, y_train).score(X_test, y_test))
            y_pred = clf.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

        if options['Classify'] == 'knn':
            print 'KNN'
            KNN = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                                       weights=best_params['weights'],
                                       leaf_size=best_params['leaf_size'])

            cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            self.my_utils.plot_learning_curve(name, KNN, 'Learning Curve (KNN)', X, y,
                                              ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(KNN, X_train, y_train)
                print('Cross Val Score Training Set (KNN): %.2f' % scores.mean())
                scores = cross_val_score(KNN, X_test, y_test)
                print('Cross Val Score Test Set (KNN: %.2f' % scores.mean())

            # ACCURACY
            print('KNN Score: %f' % KNN.fit(X_train, y_train).score(X_test, y_test))
            y_pred = KNN.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

        #if options['Classify'] == 'svm':
        if options['Classify'] == 'svm' and options['Bagging']:
            print 'SVM (bagging)'
            SVM = BaggingClassifier(svm.SVC(kernel=best_params['kernel'],
                                            degree=best_params['degree']),
                                         max_samples=0.5, max_features=0.5)

            if self.graph_lc:
                cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
                self.my_utils.plot_learning_curve(name, SVM, 'Learning Curve (SVM)', X, y,
                                                  ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(SVM, X_train, y_train)
                print('Cross Val Score Training Set (SVM): %.2f' % scores.mean())
                scores = cross_val_score(SVM, X_test, y_test)
                print('Cross Val Score Test Set (SVM: %.2f' % scores.mean())

            # ACCURACY
            print('SVM Score: %f' % SVM.fit(X_train, y_train).score(X_test, y_test))
            y_pred = SVM.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

        elif options['Classify'] == 'svm' and not options['Bagging']:
            print 'SVM'
            SVM = svm.SVC(kernel=best_params['kernel'],
                          degree=best_params['degree'])
                          #coef0=best_params['coef0'])

            cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            self.my_utils.plot_learning_curve(name, SVM, 'Learning Curve (SVM)', X, y,
                                              ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(SVM, X_train, y_train)
                print('Cross Val Score Training Set (SVM): %.2f' % scores.mean())
                scores = cross_val_score(SVM, X_test, y_test)
                print('Cross Val Score Test Set (SVM: %.2f' % scores.mean())

            # ACCURACY
            print('SVM Score: %f' % SVM.fit(X_train, y_train).score(X_test, y_test))
            y_pred = SVM.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)

        if options['Classify'] == 'nrl_net':
            print 'Neural Network'
            NRL_NET = MLPClassifier(activation=best_params['activation'],
                                    solver=best_params['solver'],
                                    learning_rate=best_params['learning_rate'])

            cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            self.my_utils.plot_learning_curve(name, NRL_NET, 'Learning Curve (NRL_NET)', X, y,
                                              ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            if self.more_stats:
                scores = cross_val_score(NRL_NET, X_train, y_train)
                print('Cross Val Score Training Set (NRL NET): %.2f' % scores.mean())
                scores = cross_val_score(NRL_NET, X_test, y_test)
                print('Cross Val Score Test Set (NRL NET: %.2f' % scores.mean())

            # ACCURACY
            print('SVM Score: %f' % NRL_NET.fit(X_train, y_train).score(X_test, y_test))
            y_pred = NRL_NET.predict(X_test)
            self.my_utils.Statistics(y_test, y_pred)


    def Find_Best_HyperP(self, X_train, y_train,
                         name, options):

        best_params = {}

        if options['Classify'] == 'd_tree':
            hyper_params = {"criterion": ["gini", "entropy"],
                            "min_samples_split": [2, 10, 20, 100, 1000],
                            "max_depth": [None, 2, 5, 10, 100],
                            "min_samples_leaf": [1, 5, 10],
                            "max_leaf_nodes": [None, 5, 10, 20]}

            # Get best params from cross validation from grid of hyper params
            # The 'else' conditional is for comparing tuned hyper params using
            # cross validation vs un-tuned, hard coded params
            if options['x_val']:
                best_params = self.my_utils.Grid_Search(DecisionTreeClassifier(), name,
                                                        X_train, y_train, hyper_params,
                                                        options['num_folds'])
            else:
                best_params = {'criterion': 'entropy',
                               'max_depth': None,
                               #'max_depth': 3,
                               'min_samples_split': 2,
                               #'min_samples_split': 10,
                               #'min_samples_split': 50,
                               #'random_state': None,
                               'min_samples_leaf': 1,
                               'max_leaf_nodes': None}

        if options['Classify'] == 'knn':
            #######################
            # KNeighborsClassifier
            # http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
            #######################
            hyper_params = {"n_neighbors": [int(0.10*np.shape(X_train)[0]),
                                            int(0.25*np.shape(X_train)[0]),
                                            int(0.50*np.shape(X_train)[0]),
                                            int(0.75*np.shape(X_train)[0])],
                            "weights": ["uniform", "distance"],
                            "leaf_size": [int(0.10*np.shape(X_train)[0]),
                                          int(0.50*np.shape(X_train)[0]),
                                          int(0.75*np.shape(X_train)[0])]}

            if options['x_val']:
                best_params = self.my_utils.Grid_Search(KNeighborsClassifier(), name,
                                                        X_train, y_train, hyper_params,
                                                        options['num_folds'])
            else:
                best_params = {'n_neighbors': 5,
                               'weights': "uniform",
                               'leaf_size': 30}

        if options['Classify'] == 'svm':
            #######################
            # svm.SVC
            # http://scikit-learn.org/stable/modules/svm.html#classification
            #######################
            hyper_params = {"kernel": ["linear", "sigmoid", "rbf", "poly"],
                            #"degree": [2, 3],
                            "degree": [2]}
                            #"coef0": [0.0, 0.1, 0.5]}

            if options['x_val']:
                best_params = self.my_utils.Grid_Search(svm.SVC(), name,
                                                        X_train, y_train, hyper_params,
                                                        options['num_folds'])
            else:
                best_params = {#'kernel': "rbf",
                               'kernel': "poly",
                               'degree': 4}   #Using deg > 2 for poly fit requires  'bagging = True' for large data sets
                               #'coef0': 0.0}

        if options['Classify'] == 'nrl_net':
            #######################
            # sklearn.neural_network.MLPClassifier
            # http://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
            #######################
            hyper_params = {"activation": ["identity", "logistic", "tanh", "relu"],
                            "solver": ["lbfgs", "sgd", "adam"],
                            "learning_rate": ["constant", "invscaling", "adaptive"]}
                            #"max_iter": [1000]}

            if options['x_val']:
                best_params = self.my_utils.Grid_Search(MLPClassifier(), name,
                                                        X_train, y_train, hyper_params,
                                                        options['num_folds'])
            else:
                best_params = {'activation': "relu",
                               'solver': "adam",
                               'learning_rate': "constant"}
                               #'max_iter': 200}

        if not options['x_val']:
            print 'Default hyper-params:', best_params
        return best_params


def main():
    my_utils = Utilities()
    my_data = SupervisedLearning()
    normalize_data = True   # Normalize the data
    x_val = True      #Perform cross validation
    num_folds = 10   #for cross validation

    #######################
    # Params and options for decision tree
    #######################
    options = {'Classify': 'd_tree', 'Bagging': True, 'Boosted': True, 'x_val': x_val, 'num_folds': num_folds}
    #options = {'Classify': 'knn', 'Bagging': True, 'Boosted': True, 'x_val': x_val, 'num_folds': num_folds}
    #options = {'Classify': 'svm', 'Bagging': True, 'Boosted': True, 'x_val': x_val, 'num_folds': num_folds}
    #options = {'Classify': 'nrl_net', 'Bagging': True, 'Boosted': True, 'x_val': x_val, 'num_folds': num_folds}

    #######################
    # 1st dataset
    #from sklearn.datasets import make_gaussian_quantiles
    #X, y = make_gaussian_quantiles(n_samples=13000, n_features=10, n_classes=3, random_state=1)
    #######################
    name = 'IRIS'

    dataset = load_iris()
    featr_names  = dataset.feature_names
    class_names  = dataset.target_names
    X = dataset.data
    y = dataset.target

    print '\n========================='
    print name
    print 'Data  Size: %d x %d' % (X.shape[0], X.shape[1])
    print '# Features: %d' % (len(featr_names))
    print '#  Classes: %d' % (len(class_names))
    print '========================='

    #######################
    # Using the train_test_split function from scikit-learn's
    # cross_validation module, randomly split the X and y arrays
    # into test data and training data
    #######################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            random_state=0)

    if normalize_data:
        X_trn_std, X_tst_std = my_utils.Standard_Scale(X_train, X_test)

        best_params = my_data.Find_Best_HyperP(X_trn_std, y_train,
                                               name, options)

        # Run best params on test set
        my_data.Classify(X, y, X_trn_std, X_tst_std, y_train, y_test,
                         name, featr_names, class_names, best_params, options)
    else:
        best_params = my_data.Find_Best_HyperP(X_train, y_train,
                                               name, options)

        # Run best params on test set
        my_data.Classify(X, y, X_train, X_test, y_train, y_test,
                         name, featr_names, class_names, best_params, options)

    #######################
    # Read in other datasets
    #######################
    #data_file = 'bank-additional.csv'
    target = 'y'   #Column 'y' is the target
    data_file = 'spamdata.csv'
    #data_file = 'OR.csv'
    name = str(data_file.split('.')[0])
    # Preprocess data
    X, y, featr_names, class_names = my_utils.Preprocess_Data(name, target, data_file, normalize_data)
    

    #######################
    # Using the train_test_split function from scikit-learn's
    # cross_validation module, randomly split the X and y arrays
    # into test data and training data
    #######################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            random_state=0)

    if normalize_data:
        X_trn_std, X_tst_std = my_utils.Standard_Scale(X_train, X_test)

        best_params = my_data.Find_Best_HyperP(X_trn_std, y_train,
                                               name, options)

        # Run best params on test set
        my_data.Classify(X, y, X_trn_std, X_tst_std, y_train, y_test,
                         name, featr_names, class_names, best_params, options)
    else:
        best_params = my_data.Find_Best_HyperP(X_train, y_train,
                                               name, options)

        # Run best params on test set
        my_data.Classify(X, y, X_train, X_test, y_train, y_test,
                         name, featr_names, class_names, best_params, options)

    #######################
    # Decision Tree Logical OR
    name = 'DT_OR'
    X = [[1, 1], [1, 2], [2, 1], [2, 2]]
    Y = [-1, 1, 1, 1]
    #class_names = map(str, list(set(Y)))
    class_names = ['-', '+']
    featr_names = ['X1', 'X2']
            
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(X, Y)
    my_utils.Plot_Decision_Tree(name, featr_names, class_names, clf)
    #######################


if __name__ == '__main__':
    main()

