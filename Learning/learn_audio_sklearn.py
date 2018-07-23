from Loading.load_songs import *
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
import sklearn
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import sys
import glob
import datetime
import matplotlib
import matplotlib.cm as cm
from textwrap import wrap
import scipy.stats as stats
from scipy import interp

# This python script will take pre-made python dictionaries with feature information about an artists songs, and do machine learning and data visualisation. When I load in songs and extract features in the other script, the data is stored in a nested dictinoary structure.

# function to load in data from load_songs script.


def prepare_data(all_data_in):
    all_features = []
    all_artists = []
    for artist in all_data_in:  # As i did feature extraction on each artist seperately, loop through them. Create lists of song names and features
        data = load_obj(artist.replace('.pkl', ''))
        print('loading {0}'.format(artist))
        songname = []  # will be a list of song names
        songfeat = []  # will be a list of dictionaries containing the song feature data
        # will be a list of artists, as I load them artist at a time should be straightforward.
        artists = []
        for song in data:  # data is a dictionary, keys are song names
            # data corresponding to each dictionary key is another dict with features and labels
            songfeat.append(data[song])
            songname.append(song)
            artists.append(artist.replace('_data.pkl', '').replace('all_', '').replace(
                path, '').replace('_data_testsplit.pkl', '').replace('_data_trainsplit.pkl', ''))
            #######################################################
            nan_keys = dict()
            items = data[song].items()
            for elem in items:
                if elem[1] is None or np.isnan(elem[1]) or np.isinf(elem[1]):
                    nan_keys.update({elem[0]: elem[1]})
            if nan_keys:
                print(song)
                print(nan_keys)

            #######################################################
        # if we want to modify the features, could do it here
        # e.g. removing some features
        '''
        print (len(songfeat[0]))
        for i in range(len(songfeat)):
            for k in ['onset_a','onset_std','bpm','centroid_a','centroid_std','polyfeat_a','polyfeat_std','zcr_a','zcr_std']:
                songfeat[i].pop(k,None)

        print (len(songfeat[0]))
        '''
        feature_names = list(
            songfeat[0].keys())  # will be all our feature names
        features = []  # will be all our raw feature data
        for i in range(len(songfeat)):
            # take the songfeat dictionary and grab only the values (keys are just text labels for each feature)
            features.append(list(songfeat[i].values()))

        # create master lists of features and artists for the machine learning later
        all_features += features
        all_artists += artists
    return all_features, all_artists, feature_names


"""    
We want to produce a chart showing the mean prediction probabilities
for each class. This function plots a confusion-matrix-like chart by
averaging over all samples in the class, the probabilities of the sample
being a member of each class according to the Random Forest.
"""
def plot_probability_matrix(test_data, predicted_data, figure=None):
    classes = np.unique(test_data)
    matrix = np.zeros(shape=(len(classes), len(classes)))
    # loop over each class
    for i in range(len(classes)):
        # locate classes in test_data and match indices to predicted_data
        class_name = classes[i]
        indices = np.where(np.asarray(test_data) == class_name)
        class_probs = predicted_data[indices]
        # average over all samples in each class and average the prediction percentage
        matrix[:, i] = np.mean(class_probs, axis=0)
    if figure is not None:
        # plot a confusion-matrix-like chart
        figure.add_subplot(2,2,1)
        plt.imshow(matrix,
                   interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.title("Probability matrix")
        fmt = '.2f'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
      #  plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    return matrix


# A function to compute and plot a ROC curve. Makes use of the "roc_curve"
# function provided by sklearn.
def plot_roc_curve(test_data, predicted_data, figure=None):
    classes = np.unique(test_data)
    n_classes = len(classes)
    # loop over classes
    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        class_name = classes[i]
        # get false-positive and true-positive rates from roc_curve for the class
        fpr[i], tpr[i], _ = roc_curve(
            test_data, predicted_data[:, i], pos_label=class_name)

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    # compute AUC
    roc_auc = auc(all_fpr, mean_tpr)
    print('AUC = {0}'.format(roc_auc))

    # Plot ROC (Macro)
    if figure is not None:
        figure.add_subplot(2,2,2)
        plt.plot(all_fpr, mean_tpr,
                 label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc),
                 linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class macro average ROC curve.')
        plt.legend(loc="lower right")
     #   plt.tight_layout()
    # return AUC just in case
    return roc_auc


# plot confusion matrix - code adapted from sklearn manual page
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(test_data, predicted_data, normalize=False, figure=None):
    classes = np.unique(test_data)
    # compute confusion matrix
    cm = confusion_matrix(test_data, predicted_data)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Showing normalized confusion matrix")
    else:
        print('Showing confusion matrix, without normalization')
    # print(cm)
    if figure is not None:
        figure.add_subplot(2,2,4)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

     #   plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


# A function to plot feature importances
def plot_feature_importances(importances, feature_names, std, figure=None, title="Feature Importances"):
    indices = np.argsort(importances)[::-1]
    # get feature names
    feature_names_importanceorder = []
    for f in range(len(indices)):
        feature_names_importanceorder.append(str(feature_names[indices[f]]))

    # Plot it yo
    if figure is not None:
        figure.add_subplot(2, 2, 3)
    else:
        plt.figure()

    plt.title(title)
    plt.bar(range(len(indices)), importances[indices], color='r', yerr=std[indices], align='center')
    plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='vertical')
    plt.xlim([-1, len(indices)])
  #  plt.tight_layout()

    return feature_names_importanceorder


# A function to save figures to a dir. The directory is autogenerated in a specified location
# using the current date & time. Figures dict should use filename : figureobject pairs.
# Also writes "meta" string containing metadata about the learning and data, to a metadata.log
# file.
def save_figs(figures_dict, meta, path_to_dir, dir_name=None):
    if dir_name is None:
        dir_name = datetime.datetime.now().isoformat()
    path = path_to_dir + '/' + dir_name
    # check if the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise ValueError("Path: ", path, " already exists!")

    # loop over dict entries
    for filename, fig in figures_dict.items():
        # set filepath for saving
        filepath = path + '/' + filename
        # switch current figure
        plt.figure(fig.number)
        # The following values adjust padding etc. of the subplots.
        # These optimise the layout for a 16:9 display with the plots arranged
        # as they are in the functions provided. The easiest way to adjust these
        # is by playing around in an interactive figure window (if possible) until
        # you get the layout you want and then copying the numbers down. Sadly
        # tight_layout makes the plots far too small to be useful.
        plt.subplots_adjust(top=0.92,
                            bottom=0.20,
                            left=0.10,
                            right=0.88,
                            hspace=0.60,
                            wspace=0.30)

        plt.savefig(filepath, bbox_inches='tight')

    # save metadata
    with open(path + '/' + 'metadata.log', 'w') as f:
        f.write(meta)
    # inform the user
    print("Data written to ", path)

# Here we go, let's try some machine learning algorithms

if __name__ == '__main__':

    # Set matplotlib params
    # This changes the size of created figures. Adjust for your display if necessary
    matplotlib.rcParams['figure.figsize'] = [18, 9]
    
    # load in all data saved from the feature extraction, *.pkl. Initiate figure and select colours
    path = sys.argv[1]  # command line input is path to data
    all_data = glob.glob(path + '/*_data.pkl')  # load in as many as you want

    colors = iter(cm.Set1(np.linspace(0, 1, len(all_data))))
    #colors = iter(cm.cubehelix(np.linspace(0, 1, len(all_data))))

    # load in artists with loads of songs - may or may not be splitting songs, try except:
    # feature names is same for all runs when unpacked (saves loading in a .pkl again)
    all_features, all_artists, feature_names = prepare_data(all_data)
    # Split our data into a training and testing data sets
    train_percent = float(sys.argv[2])
    # Test/train split as usual on artists with many songs
    features_train, features_test, artists_train, artists_test = train_test_split(
        all_features, all_artists, train_size=train_percent, random_state=0, stratify=all_artists)

    # now data is prepared for machine learning
    try:
        if len(artists_test) == len(features_test) and len(artists_train) == len(features_train):
            None
    except:
        print('artists and features are not same length: {0} != {1}', format(
            artists_test, features_test, artists_train, features_train))
        sys.exit()

    feature_names_flatten = np.array(feature_names).flatten()
    feature_names = np.transpose(feature_names)
    # print(np.transpose(feature_names))
    # set up data as numerical classes as well as labeled string classes - some classifiers require numbered labels, not artists as strings
    X_test = np.array(features_test)
    Y_test = np.array(artists_test)
    le = preprocessing.LabelEncoder()
    le.fit(Y_test)
    Y_test_n = le.transform(Y_test)

    X_train = np.array(features_train)
    Y_train = np.array(artists_train)
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y_train_n = le.transform(Y_train)

    names = np.unique(Y_test)
    print(names)
    # Now try some classifiers!

    # Set up neutral network classifier

    from sklearn.preprocessing import StandardScaler
    nn_features_train = X_train
    nn_features_test = X_test
    scaler = StandardScaler()
    # normalise data and remove mean:
    scaler.fit(features_train)
    nn_features_train = scaler.transform(nn_features_train)
    nn_features_test = scaler.transform(nn_features_test)

    from sklearn.neural_network import MLPClassifier
    # not sure how many nodes to use? loop over some values until you're sure you've maxed the accuracy- 5000 is good for this:
    for i in range(5000, 5001, 1000):
        nn = MLPClassifier(hidden_layer_sizes=(
            i, ), solver='adam', max_iter=2000)
        nn.fit(nn_features_train, artists_train)
        nn_pred = nn.predict(nn_features_test)
        # get classification report
        nn_report = ('--'*30 + '\n'
                     'MLP nn classifier with {0} hidden layers'.format(i) + '\n'
                     '{0}'.format(classification_report(artists_test, nn_pred, target_names=names)) + '\n'
                     + '--'*30 + '\n'
                     )
        print(nn_report)
       
    '''
    #we could try SVC; it's quite poor compared to random forests.
    clf = svm.SVC(class_weight='balanced')
    clf.fit(X_train, Y_train_n)
    artists_SVM_pred=clf.predict(X_test)
    print('--'*30)
    print('SVM report:')
    print(classification_report(Y_test_n, artists_SVM_pred,target_names=names))
    print('--'*30)
    '''

    # Build a forest and compute the feature importances
    n_estimators = 2000  # number of trees?
    forest = RandomForestClassifier(
        n_estimators=n_estimators, random_state=2, class_weight='balanced')
    forest.fit(features_train, artists_train)
    artists_pred = forest.predict(features_test)
    artists_proba = forest.predict_proba(features_test)
    # we'll print this later as a comparison
    accuracy_before = (accuracy_score(artists_test, artists_pred))

    # Could check classification report
    forest_unpruned_report = ('--'*30 + '\n'
                              'Random forest before pruning:\n'
                              '{0}'.format(classification_report(artists_test, artists_pred, target_names=names)) + '\n'
                              + '--'*30 +
                              '\n'
                              )
    print(forest_unpruned_report)

    # Plots before pruning!
    fig_unpruned = plt.figure()
    plot_probability_matrix(artists_test, artists_proba, figure=fig_unpruned)
    plot_roc_curve(artists_test, artists_proba, figure=fig_unpruned)
    plot_confusion_matrix(artists_test, artists_pred, figure=fig_unpruned)

    # plot importances unpruned
    
    importances_unpruned = forest.feature_importances_
    std_unpruned = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices_unpruned = np.argsort(importances_unpruned)[::-1]
    
    title_unpruned = """n_est={0}, train-test={1}%, Accuracy={2:.3f}""".format(n_estimators, train_percent*100, accuracy_before, 40)
    plot_feature_importances(importances_unpruned, feature_names, std_unpruned, figure=fig_unpruned, title=title_unpruned)

    
    
    # you could loop over trees to find out how many before you accuracy maxes output
    '''
    #how many trees are required? loop through values to find out
    scores=[]
    for val in range(1,100,10): #set accordingly
        clf=RandomForestClassifier(n_estimators=val,class_weight='balanced')
        validated = cross_validate(clf,X,Y,cv=5,scoring=['f1_weighted'])
        scores.append(validated)

    #make a nice plot:
    for i in range(0,len(scores)):
        print(scores[i]['test_f1_weighted'])
    y=[]
    x=[]
    e=[]
    for i in range(0,len(scores)):
        x.append(i)
        y.append(np.mean(scores[i]['test_f1_weighted']))
        e.append(np.mean(np.std(scores[i]['test_f1_weighted'])))
        print(np.mean(scores[i]['test_f1_weighted']), np.std(scores[i]['test_f1_weighted']))
    plt.errorbar(x,y,e)
    plt.show()
    '''

    ##############################################################################
    # Now lets repeat using a more streamlined pipeline to first remove unimportant features, then run a classifier on remaining ones.
    # we may want to try different classifiers and feature selection processes
    # an important note is that the pipeline automatically creates new feature data after removing pruned features.

    # first choose a model to prune features, then put it in pipeline - there are many we could try
    feature_selection_threshold = 0.1
    lsvc = LinearSVC(C=feature_selection_threshold, penalty="l1", dual=False).fit(
        features_train, artists_train)
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=2)
    modelselect = 'lsvc'  # set accordingly
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(lsvc)),
        ('classification', RandomForestClassifier(
            n_estimators=n_estimators, random_state=2, class_weight='balanced'))
    ])
    # do the fit and feature selection
    pipeline.fit(features_train, artists_train)
    # check accuracy and other metrics:
    artists_important_pred = pipeline.predict(features_test)
    artists_important_proba = pipeline.predict_proba(features_test)
    accuracy_after = (accuracy_score(artists_test, artists_important_pred))

    forest_pruned_report = ('accuracy before pruning features: {0:.2f}'.format(accuracy_before) + '\n'
                            'accuracy after pruning features: {0:.2f}'.format(accuracy_after) + '\n'
                            + '--' * 30 + '\n'
                            'Random Forest report after feature pruning:\n'
                            '{0}'.format(classification_report(artists_test, artists_important_pred, target_names=names)) + '\n'
                            + '--' * 30 + '\n'
                            'Log-loss = {0}'.format(log_loss(artists_test, artists_important_proba)) + '\n'
                            )
    print(forest_pruned_report)

    # Declare a figure for plotting all subplots on plot probability matrix,
    # roc curve, and confusion matrix. (The order of subplots is established
    # inside each function that takes "fig" as an argument
    fig_pruned = plt.figure()
    plot_probability_matrix(artists_test, artists_important_proba, figure=fig_pruned)
    plot_roc_curve(artists_test, artists_important_proba, figure=fig_pruned)
    plot_confusion_matrix(artists_test, artists_important_pred, figure=fig_pruned)

    
    # Now make get feature importances with standard deviations
    clf = pipeline.steps[1][1]  # get classifier used
    importances_pruned = pipeline.steps[1][1].feature_importances_
    std_pruned = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices_pruned = np.argsort(importances_pruned)[::-1]
    # Construct title
    title_long = """Features pruned by {0}. n_est={1}, train-test={2}%
    , Accuracy - before={3:.3f}, after={4:,.3f}""".format(
        modelselect, n_estimators, train_percent*100, accuracy_before, accuracy_after, 40)
    # grab the pruned feature names from plot_feature_importances, and plot the chart
    feature_names_importanceorder_pruned = plot_feature_importances(importances_pruned, feature_names, std_pruned, figure=fig_pruned, title=title_long)

    # see which features were removed
    no_features = len(feature_names_importanceorder_pruned)
    print('Started with {0} features, now using {1}'.format(
        len(feature_names), no_features))
    print('features used were:')
    print(set(feature_names_flatten) - set(feature_names_importanceorder_pruned))

    #??? np.set_printoptions(precision=2)
    

    # plt.figure(fig_pruned.number)
    # plt.tight_layout()

    # create metadata
    meta = ('music-ML metadata.log\n'
            + '=='*50 + '\n'
            'Classes:\n{0}'.format(np.unique(artists_test)) + '\n'
            + '=='*50 + '\n'
            'Features:\n{0}'.format(feature_names) + '\n'
            + '=='*50 + '\n'
            'Classification Reports:\n\n'
            '{0}\n\n{1}\n\n{2}\n\n'.format(nn_report, forest_unpruned_report, forest_pruned_report)
            )

    # construct figures dict
    figures_dict = {'rnd_forest_unpruned_graphs': fig_unpruned, 'rnd_forest_pruned_graphs': fig_pruned}
    savedir = '/raid/scratch/sen/learning_results/sklearn/LSVC/'
    dir_name = '{0}_selection_threshold'.format(feature_selection_threshold)
    save_figs(figures_dict, meta, savedir, dir_name)

    plt.show()
