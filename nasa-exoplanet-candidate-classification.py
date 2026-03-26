# import relevant modules
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import pre and post ml packages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import permutation_importance

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def main():

    # gather data
    file_name="cumulative_2023.11.07_13.44.30 (1).csv"
    df=pd.read_csv(file_name,skiprows=41)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # select attributes
    drop_columns= [k for k in df.columns if "err" in k] + ["koi_disposition"]
    df=df.drop(columns=drop_columns)

    # drop NA values
    df=df.dropna()

    # split into feature and target vector
    X=df.drop(columns="koi_pdisposition")
    y=df["koi_pdisposition"]

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.75,random_state=10,stratify=y)

    # use standard scalar to standardize training and testing data
    tfr_standardscale = StandardScaler()
    X_tfr=tfr_standardscale.fit_transform(X_train)
    X_train_tfr = pd.DataFrame(X_tfr, columns=X_train.columns)
    X_test_tfr = pd.DataFrame(tfr_standardscale.transform(X_test), columns=X_test.columns)

    # build pipeline
    pipe=Pipeline([
        ('estimator',None)
    ])

    # set all hyperparam ranges
    n_neighbors_range=range(1,1+int(1.5*math.sqrt(X_train.shape[0])))
    criterion_range=["entropy","gini"]
    max_depth_range=range(4,22)
    min_samples_leaf_range=range(1,9)
    kernel_range=["rbf"]
    C_range=[0.01,0.1,1,10]
    gamma_range=[0.1,1,10]
    hidden_layer_sizes_range=[(20,),(20,20),(20,20,20)]
    max_iter_range=[100]

    # set iteration numbers in randomized search to make sure searching at least 10% of the total hyperparam space
    iterations=int(0.1*(1+len(n_neighbors_range)+len(criterion_range)*len(max_depth_range)*len(min_samples_leaf_range)+len(kernel_range)*len(C_range)*len(gamma_range)+len(hidden_layer_sizes_range)))

    # create param distribution
    estimator_list=[{
        'estimator':[LogisticRegression()]
    },{
        'estimator':[KNeighborsClassifier()],
        'estimator__n_neighbors':n_neighbors_range
    },{
        'estimator':[DecisionTreeClassifier()],
        'estimator__criterion':criterion_range,
        'estimator__max_depth':max_depth_range,
        'estimator__min_samples_leaf':min_samples_leaf_range
    },{
        'estimator':[SVC()],
        'estimator__kernel':kernel_range,
        'estimator__C':C_range,
        'estimator__gamma':gamma_range
    },{
        'estimator':[MLPClassifier()],
        'estimator__hidden_layer_sizes':hidden_layer_sizes_range,
        'estimator__max_iter':max_iter_range
    }]

    # use randomized search to cross-validate models via pipeline
    rmcv_npca=RandomizedSearchCV(
        pipe,
        param_distributions=estimator_list,
        n_iter=iterations,
        scoring="accuracy"
    )

    # fit non-pca data into rmcv
    rmcv_npca.fit(X_train_tfr,y_train)

    # extract best params and best scores for non-pca case
    best_params_npca=rmcv_npca.best_params_
    best_scores_npca=rmcv_npca.best_score_

    # PCA process
    pca=PCA(n_components=0.95)
    X_train_pca=pd.DataFrame(pca.fit_transform(X_train_tfr),index=X_train_tfr.index)
    X_test_pca=pd.DataFrame(pca.transform(X_test_tfr),index=X_test_tfr.index)

    # fit pca data into rmcv
    rmcv_pca=rmcv_npca # essentially contains the same arguments with non-pca case, use different variable name to differentiate
    rmcv_pca.fit(X_train_pca,y_train)

    # extract best params and best scores for pca case
    best_params_pca=rmcv_pca.best_params_
    best_scores_pca=rmcv_pca.best_score_

    # determine better model, params and corresponding training data
    if best_scores_pca >= best_scores_npca:
        X_train_best=X_train_pca
        X_test_best=X_test_pca
        params_best=best_params_pca
        q1_response="PCA data performs better. PCA improve upon the results from when I didn't use it."   # determine whether pca data performs better or not. Not used in final output.
    else:
        X_train_best=X_train_tfr
        X_test_best=X_test_tfr
        params_best=best_params_npca
        q1_response="Non-PCA data performs better. PCA did NOT improve upon the results from when I didn't use it."
      
    print(q1_response)
    
    # extract params to use for grid search
    model_best=params_best['estimator']
    param_grid_best = [d for d in estimator_list if d["estimator"]==[model_best]]

    # set up grid search cv
    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid_best,
        scoring="accuracy")

    # train model
    gscv.fit(X_train_best, y_train)

    # use branching to setup final optimal model for training
    if model_best == KNeighborsClassifier():
        best_n_neighbors=gscv.best_params_['estimator__n_neighbors']
        final_model=KNeighborsClassifier(n_neighbors=best_n_neighbors)
    elif model_best == DecisionTreeClassifier():
        best_criterion=gscv.best_params_['estimator__criterion']
        best_max_depth=gscv.best_params_['estimator__max_depth']
        best_min_samples_leaf=gscv.best_params_['estimator__min_samples_leaf']
        final_model=DecisionTreeClassifier(criterion=best_criterion,max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf)
    elif model_best == SVC():
        best_kernel=gscv.best_params_['estimator__kernel']
        best_C=gscv.best_params_['estimator__C']
        best_gamma=gscv.best_params_['estimator__gamma']
        final_model=SVC(kernel=best_kernel,C=best_C,gamma=best_gamma)
    elif model_best == MLPClassifier():
        best_hidden_layer_sizes=gscv.best_params_['estimator__hidden_layer_sizes']
        best_max_iter=gscv.best_params_['estimator__max_iter']
        final_model=MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes,max_iter=best_max_iter)
    else:
        final_model=LogisticRegression()
    
    # train final model
    final_model.fit(X_train_best,y_train)

    # get target vector predictions
    y_pred=final_model.predict(X_test_best)

    # create and confusion matrix display object
    cm=confusion_matrix(y_test, y_pred, labels=final_model.classes_)
    cm_disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=final_model.classes_)

    # print classification report
    class_rep = classification_report(y_test, y_pred)
    print("Classification report:")
    print(class_rep)

    # create figure, plot confusion matrix, customize figure, and save figure
    fig,ax=plt.subplots(1,1,figsize=(16,10),dpi=300)
    cm_disp.plot(ax=ax)
    ax.set(title="Optimal Confusion Matrix")
    fig.tight_layout()
    fig.savefig("Confusion Matrix Visualization.png")
