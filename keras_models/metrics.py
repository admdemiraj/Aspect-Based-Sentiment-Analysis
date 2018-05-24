from keras import backend as K
from utilities import load_model_JSON
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import os


def f12(y_true, y_pred):
    y_true = K.eval(y_true)
    y_pred = K.eval(y_pred)
    return f1_score(y_true, y_pred)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def accuracy(y_true, y_predicted):
    ''' Typical calculation of the accuracy for eaach of the categories'''
    total_predictions = len(y_true)
    print("total predictions", total_predictions)
    correct_predictions = 0
    wrong_predictions = 0
    correct_predictions_cat0 = 0
    correct_predictions_cat1 = 0
    correct_predictions_cat2 = 0
    correct_predictions_cat3 = 0
    correct_predictions_cat4 = 0
    correct_predictions_cat5 = 0
    correct_predictions_cat6 = 0
    correct_predictions_cat7 = 0
    correct_predictions_cat8 = 0
    correct_predictions_cat9 = 0
    correct_predictions_cat10 = 0
    correct_predictions_cat11 = 0
    tp = 0
    fp = 0
    fn = 0
    for i in range(0,len(y_true)):
        if np.equal(y_true[i],y_predicted[i]).all():
            correct_predictions += 1
            # check in which category are the predictions
        else:
            wrong_predictions += 1
        for j in range(0,11):
            if y_true[i][j] == y_predicted[i][j] and y_true[i][j] == 1:
                tp += 1
            elif y_true[i][j] != y_predicted[i][j] and y_true[i][j] == 1:
                fn += 1
            elif y_true[i][j] != y_predicted[i][j] and y_true[i][j] == 0:
                fp += 1

        if y_true[i][0] == 1 and y_predicted[i][0] == 1:
            correct_predictions_cat0 += 1
        elif y_true[i][1] == 1 and y_predicted[i][1] == 1:
            correct_predictions_cat1 += 1
        elif y_true[i][2] == 1 and y_predicted[i][2] == 1:
            correct_predictions_cat2 += 1
        elif y_true[i][3] == 1 and y_predicted[i][3] == 1:
            correct_predictions_cat3 += 1
        elif y_true[i][4] == 1 and y_predicted[i][4] == 1:
            correct_predictions_cat4 += 1
        elif y_true[i][5] == 1 and y_predicted[i][5] == 1:
            correct_predictions_cat5 += 1
        elif y_true[i][6] == 1 and y_predicted[i][6] == 1:
            correct_predictions_cat6 += 1
        elif y_true[i][7] == 1 and y_predicted[i][7] == 1:
            correct_predictions_cat7 += 1
        elif y_true[i][8] == 1 and y_predicted[i][8] == 1:
            correct_predictions_cat8 += 1
        elif y_true[i][9] == 1 and y_predicted[i][9] == 1:
            correct_predictions_cat9 += 1
        elif y_true[i][10] == 1 and y_predicted[i][10] == 1:
            correct_predictions_cat10 += 1
        elif y_true[i][11] == 1 and y_predicted[i][11] == 1:
            correct_predictions_cat11 += 1

    precision = tp/(tp+fp)
    recal = tp/(tp+fn)
    f1_ = 2*precision*recal/(precision+recal)
    print("As calculated : Precison :", precision, "\nRecall: ", recal, "\nf1", f1_)
    print("correct predictions: ", correct_predictions)
    print("wrong predictions: ", wrong_predictions)
    print("accuracy is: ", correct_predictions/total_predictions)
    print("accuracy in cat0 is: ", correct_predictions_cat0 / 142)
    print("accuracy in cat1 is: ", correct_predictions_cat1 / 21)
    print("accuracy in cat2 is: ", correct_predictions_cat2 / 33)
    print("accuracy in cat3 is: ", correct_predictions_cat3 / 22)
    print("accuracy in cat4 is: ", correct_predictions_cat4 / 226)
    print("accuracy in cat5 is: ", correct_predictions_cat5 / 48)
    print("accuracy in cat6 is: ", correct_predictions_cat6 / 3)
    print("accuracy in cat7 is: ", correct_predictions_cat7 / 21)
    print("accuracy in cat8 is: ", correct_predictions_cat8 / 12)
    print("accuracy in cat9 is: ", correct_predictions_cat9 / 57)
    print("accuracy in cat10 is: ", correct_predictions_cat10 / 145)
    print("accuracy in cat11 is: ", correct_predictions_cat11 / 13)


def accuracy2(y_true, y_predicted):
    ''' Typical calculation of the accuracy for each category'''
    total_predictions = len(y_true)
    print("total predictions", total_predictions)
    print('number of instances per category test: ', np.sum(y_true, axis=0))
    correct_predictions = 0
    wrong_predictions = 0
    correct_predictions_cat0 = 0
    correct_predictions_cat1 = 0
    correct_predictions_cat2 = 0
    for i in range(0, len(y_true)):
        # check if the provided answer is correct
        if np.equal(y_true[i], y_predicted[i]).all():
            correct_predictions += 1
            # check in which category are the predictions
            if(y_true[i][0] == 1):
                correct_predictions_cat0 += 1
            elif(y_true[i][1] == 1):
                correct_predictions_cat1 += 1
            else:
                correct_predictions_cat2 += 1

        else:
            wrong_predictions += 1

    print("correct predictions: ", correct_predictions)
    print("wrong predictions: ", wrong_predictions)
    print("accuracy is: ", correct_predictions/total_predictions)
    print("accuracy in cat0 is: ", correct_predictions_cat0/611)
    print("accuracy in cat1 is: ", correct_predictions_cat1/248)
    print("accuracy in cat2 is: ", correct_predictions_cat2/44)

    return correct_predictions/total_predictions


def voting_systems_polarity(y0, y1, y2, y3, y4):
    '''
    Method that takes as input 5 different predictions(numpy arrays) from the same system with different seeds
    compares the results and predicts as the correct label the one that is voted by the majority
    (at least 3 out of 5 systems should vote for it)
    :param y1:
    :param y2:
    :param y3:
    :param y4:
    :param y5:
    :return:
    '''
    final_score = np.zeros_like(y0)
    for i in range(0, y0.shape[0]):
        # find the indexes of 1
        vote0 = np.argmax(y0[0] == 1)
        vote1 = np.argmax(y1[0] == 1)
        vote2 = np.argmax(y2[0] == 1)
        vote3 = np.argmax(y3[0] == 1)
        vote4 = np.argmax(y4[0] == 1)

        # keep in a dictionary which system voted which category
        dict = {}
        dict[vote0] = y0[i]
        dict[vote1] = y1[i]
        dict[vote2] = y2[i]
        dict[vote3] = y3[i]
        dict[vote4] = y4[i]

        sum = vote0+vote1+vote2+vote3+vote4

        def voter(cat, all_votes):
            if cat == "pos":
                return all_votes[0]
            elif cat == "neg":
                return all_votes[1]
            else:
                return all_votes[2]
        # if sum is less than 2 means that most systems voted category 0 (positive).
        # Accordingly for the other two categories
        if sum <= 2:
            final_score[i] = voter("pos", dict)
        elif sum < 6:
            final_score[i] = voter("neg", dict)
        elif sum >= 6:
            final_score[i] = voter("neut", dict)

    return final_score


def voting_systems_entity_attribute(y0, y1, y2, y3, y4):
    '''
    Method that takes as input 5 different predictions(numpy arrays) from the same system with different seeds
    compares the results and predicts as the correct label the one that is voted by the majority
    (at least 3 out of 5 systems should vote for it)
    :param y1:
    :param y2:
    :param y3:
    :param y4:
    :param y5:
    :return:
    '''
    final_score = np.zeros_like(y0)
    for i in range(0, y0.shape[0]):
        # find the indexes of 1
        votes = []
        votes.append(np.argmax(y0[i] == 1))
        votes.append(np.argmax(y1[i] == 1))
        votes.append(np.argmax(y2[i] == 1))
        votes.append(np.argmax(y3[i] == 1))
        votes.append(np.argmax(y4[i] == 1))

        # keep in a dictionary which system voted which category
        dict = {}
        dict[votes[0]] = y0[i]
        dict[votes[1]] = y1[i]
        dict[votes[2]] = y2[i]
        dict[votes[3]] = y3[i]
        dict[votes[4]] = y4[i]
        # which is the category that won
        winner = 0
        total_votes = 0
        for j in range(0, len(votes)):
            # how many systems voted for a specific category
            counter = 0
            for k in range(j+1, len(votes)):
                if votes[j] == votes[k]:
                    counter += 1
            if counter > total_votes:
                total_votes = counter
                winner = votes[j]


        final_score[i] = dict[winner]

    return final_score


def evaluate_saved_models_slot3(X_test, gold_aux, y_test, gold_review):
    ''' Check the performance of the already saved models'''
    # get current directory
    path = os.getcwd()
    # get one directory up
    path = os.path.dirname(path)
    # go to the saved models folder
    path = path + "/saved_models/slot3/model"

    model0 = load_model_JSON((str(0) + "_augmented"), path=path, f1=f1)
    model1 = load_model_JSON((str(1) + "_augmented"), path=path, f1=f1)
    model2 = load_model_JSON((str(2) + "_augmented"), path=path, f1=f1)
    model3 = load_model_JSON((str(3) + "_augmented"), path=path, f1=f1)
    model4 = load_model_JSON((str(4) + "_augmented"), path=path, f1=f1)
    scores0 = model0.predict([X_test, gold_aux])
    scores1 = model1.predict([X_test, gold_aux])
    scores2 = model2.predict([X_test, gold_aux])
    scores3 = model3.predict([X_test, gold_aux])
    scores4 = model4.predict([X_test, gold_aux])

    def turn_largest_value_to_1(scores):
        # turn the largest value of the result to 1 and the rest to 0
        # e.g. ginen [0.4, 0.3, 0.3] output [1, 0, 0]
        scores2 = np.zeros_like(scores)
        scores2[np.arange(len(scores)), scores.argmax(1)] = 1
        return scores2

    print("system: 0 ----------------------------------------->")
    y0 = turn_largest_value_to_1(scores0)
    acc = accuracy2(y_test, y0)
    print("system: 1 ----------------------------------------->")
    y1 = turn_largest_value_to_1(scores1)
    acc = accuracy2(y_test, y1)
    print("system: 2 ----------------------------------------->")
    y2 = turn_largest_value_to_1(scores2)
    acc = accuracy2(y_test, y2)
    print("system: 3 ----------------------------------------->")
    y3 = turn_largest_value_to_1(scores3)
    acc = accuracy2(y_test, y3)
    print("system: 4 ----------------------------------------->")
    y4 = turn_largest_value_to_1(scores4)
    acc = accuracy2(y_test, y4)

    scores2 = voting_systems_polarity(y0, y1, y2, y3, y4)
    print("voting system:------------------------------------->")
    acc = accuracy2(y_test, scores2)


def evaluate_saved_models_slot1(X_test, y_test):
    # get current directory
    path = os.getcwd()
    # get one directory up
    path = os.path.dirname(path)
    # go to the saved models folder
    path = path + "/saved_models/slot1/model"

    model0 = load_model_JSON(str(0), path=path, f1=f1)
    model1 = load_model_JSON(str(1), path=path, f1=f1)
    model2 = load_model_JSON(str(2), path=path, f1=f1)
    model3 = load_model_JSON(str(3), path=path, f1=f1)
    model4 = load_model_JSON(str(4), path=path, f1=f1)
    scores0 = model0.predict(X_test)
    scores1 = model1.predict(X_test)
    scores2 = model2.predict(X_test)
    scores3 = model3.predict(X_test)
    scores4 = model4.predict(X_test)

    def turn_values_larger_than_half_to_1(scores):
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        return scores

    def evaluation_measures(y_test, preds):
        print("E#A the f1 score is: ", f1_score(y_test, preds, pos_label=1, average='micro') * 100)
        print("E#A the precision score is: ", precision_score(y_test, preds, pos_label=1, average='micro') * 100)
        print("E#A the recall score is: ", recall_score(y_test, preds, pos_label=1, average='micro') * 100)

    print("system: 0 ----------------------------------------->")
    y0 = turn_values_larger_than_half_to_1(scores0)
    y0 = np.array(y0)
    evaluation_measures(y_test, y0)
    print("system: 1 ----------------------------------------->")
    y1 = turn_values_larger_than_half_to_1(scores1)
    y1 = np.array(y1)
    evaluation_measures(y_test, y1)
    print("system: 2 ----------------------------------------->")
    y2 = turn_values_larger_than_half_to_1(scores2)
    y2 = np.array(y2)
    evaluation_measures(y_test, y2)
    print("system: 3 ----------------------------------------->")
    y3 = turn_values_larger_than_half_to_1(scores3)
    y3 = np.array(y3)
    evaluation_measures(y_test, y3)
    print("system: 4 ----------------------------------------->")
    y4 = turn_values_larger_than_half_to_1(scores4)
    y4 = np.array(y4)
    evaluation_measures(y_test, y4)

    y_voting = voting_systems_entity_attribute(y0, y1, y2, y3, y4)
    print("voting system:-------------------------------------->")
    evaluation_measures(y_test, y_voting)