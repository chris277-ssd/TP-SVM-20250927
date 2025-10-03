#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################

n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)


plt.close("all")
plt.ion()  #interactivité pourquoi ?
plt.figure(1, figsize=(15, 10))
plt.title('First data set')
plot_2d(X1, y1)
plt.show()

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure(figsize=(15,15))
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print('Meilleur C estimé : %s' % clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure(figsize=(15,15))
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#1)               Iris Dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data # [150x4] 1ere 50 lignes variétés setosa , 51 : versicolor, 101 : virginica = variable varietés
#4 colonnes, 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
y = iris.target #0 : setosa, 1: versicolor,  2: virginica
X = scaler.fit_transform(X, y=y) # toutes Les données sont centrées réduites
X = X[y != 0, :2]
y = y[y != 0] # il va donc rester: sepal length (cm)', 'sepal width (cm)' en colonne et 50 lignes versicolor puis 50 lignes virginica
#le but de la manip va être de distinguer la largeur de la longueur des sépales, especes mélangées
y = np.where(y == 1, -1, +1)   # classe 1 -> -1, classe 2 -> +1


# split train test (say 25% for the test). You can shuffle and then separate or you can just use
# train_test_split whithout shuffling (in that case fix the random state (say to 42) for reproductibility)
#With shuffle :
Xs,ys=shuffle(X,y,random_state=0) # mélange des X syncho avec celui des y et remélange :
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.25, random_state=42)

###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

#%%
# Q1 Linear kernel

plt.close("all")
plt.ion()  #interactivité pourquoi ?
plt.figure(2, figsize=(15, 10))
plt.title('Pétal et Sépal mélangés')
plot_2d(X_train, y_train)
plt.show()


# fit the model and select the best hyperparameter C
#on choisit Grid search pour trouver le Best_ C
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf3 = SVC()
clf_linear = GridSearchCV(clf3, parameters, n_jobs=-1, cv=10)
clf_linear.fit(X_train, y_train)

print('Meilleur paramètre C : %s' % clf_linear.best_estimator_.C)

# compute the score
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#%%
# Q2 polynomial kernel, fit the model and select the best set of hyperparameters
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]
clf4 = SVC()
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly =GridSearchCV(clf4, parameters, n_jobs=-1, cv=10,)
clf_poly.fit(X_train, y_train)


print('Meilleur parmètre C : %s' % clf_linear.best_estimator_.C)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


#%%
# display your results using frontiere (svm_source.py)

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel

###############################################################################
#Q3,  Jeux de données déséquilibrée 92/8 %
n3 = 500
n4 = 40
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X2, y2 = rand_bi_gauss(n3, n4, mu1, mu2, sigma1, sigma2)

plt.ion()  #interactivité pourquoi ?
plt.figure(figsize=(15, 10))
plt.title('Unbalenced data set')
plot_2d(X2, y2)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=42)

#C=1 par défaut
clf5 = SVC(kernel='linear')
clf5.fit(X_train, y_train)
clf5.fit_status_
print('Score : %s' % clf5.score(X_test, y_test))

##Comparaison de différentes valeurs de C :
# Liste des valeurs de C à tester
C_list = [0.01, 1, 100]

# Taille de la figure
plt.figure(figsize=(15, 5))

for i, C in enumerate(C_list, 1):
    # Créer et entraîner le classifieur
    clf6 = SVC(kernel='linear', C=C)
    clf6.fit(X_train, y_train)
    
    # Calculer le score / test
    score = clf6.score(X_test, y_test)
    
    # Définir la fonction prédictive personnalisée
    def f6(xx):
        return clf6.predict(xx.reshape(1, -1))

    # Sous-figure
    plt.subplot(1, len(C_list), i)
    frontiere(f6, X_train, y_train, w=None, step=50, alpha_choice=1)
    plt.title(f"C = {C} | Score = {score:.2f}")

plt.tight_layout()
plt.show()
# C contrôle la pénalité sur les erreurs de classification.
#Quand C est très petit, le modèle tolère beaucoup d’erreurs pour maximiser la marge.
#Il préfère une marge large, même si cela signifie mal classer plusieurs points.
#Autrement dit, avec C = 0.01, le SVM devient très souple : il accepte que des points
#soient mal classés pour obtenir une frontière plus “stable” ou “généreuse”.
#Si les classes ne sont pas parfaitement séparables, ou si tu as du bruit, le modèle
#peut même ignorer complètement la structure réelle des données.
# si C = 1 : pénalise les erreurs modérément.
# siC = 100 : pénalise fortement les erreurs → le modèle essaie de tout classer parfaitement.
#Mais si les données sont déjà bien séparées, alors même un C modéré suffit à tracer une frontière
#quasi parfaite. Le modèle n’a pas besoin de forcer davantage, donc la frontière reste quasi identique.

#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0) # raccourci du python, axis ?
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)
ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")

#Pour l'erreur c'est l'evenement contraire :
errors = [1 - s for s in scores]

plt.figure()
plt.plot(Cs, errors, marker='o')
plt.xscale("log")
plt.xlabel("Paramètre de régularisation C (log)")
plt.ylabel("Erreur de prédiction")
plt.title("Erreur en fonction de C")
plt.grid(True)
plt.tight_layout()
plt.show()
Cs[2]
t0 = time()

#%%
# predict labels for the X_test images with the best classifier
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)

# Prédire les labels sur le jeu de test
y_pred = clf.predict(X_test)


print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

#C petit → marge large, plus d’erreurs → erreur élevée.
#C grand → modèle rigide, moins d’erreurs (mais attention à l’overfitting).
#Le minimum de la courbe te donne le compromis optimal entre biais et variance.
#%%
# Q5

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters, cv=10)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)
# parfait sur les données d'entrainement et bon à très bon sur celles de test (50%)
print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, ) 
#with gaussian coefficients of std sigma
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy,y)
# gros impact du bruit des les images de test
#%%
# Q6
#Pas mal de reflexion. Au final, l'ACP joue un role de filtre anti-bruit et projette
# les données dans l'espace des CPs.Puis on récupère les X filtrés par transform
# pour pouvoir comparer les données entre train et test, je pense qu'il est mieux
# qu'elles aient subit le même traitement. Même si on perd en indépendance et ajoute un biais.
#fonction faisant l'ACP sur X et centrant réduisant X

# Liste des dimensions à tester
components_list = [3, 8, 15, 20, 40]
scores_train = []
scores_test = []

#definition des fonction de np
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))} 
svr = svm.SVC()
clf_lin = GridSearchCV(svr, parameters, cv=10)


for n in components_list:
    #print(n)
    pca = PCA(n_components=n, svd_solver='randomized')
    X_pca = pca.fit_transform(X_noisy)
    #print(X_pca[1,:5])
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5, random_state=None)

    # Centrage et réduction des données avant SVC, les Y sont ok
    X_train_pca = scaler.fit_transform(X_train)
    X_test_pca = scaler.fit_transform(X_test)
    #print(X_train_pca[1,:5])
    clf_lin.fit(X_train_pca, y_train)
    
    scores_train.append(clf_lin.score(X_train_pca, y_train))
    scores_test.append(clf_lin.score(X_test_pca, y_test))

# Graphique des scores
plt.figure(figsize=(8, 5))
plt.plot(components_list, scores_train, marker='o', label='Score train')
plt.plot(components_list, scores_test, marker='s', label='Score test')
plt.xscale("log")
plt.xlabel("Nombre de composantes PCA")
plt.ylabel("Score de classification")
plt.title("Impact de la réduction de dimension sur le SVM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#tableau des données pour controle des valeurs:
print(components_list,scores_train,scores_test)

#Q7 : voir compte rendu. Code plus haut. Code à corriger en fonction du temps
    
