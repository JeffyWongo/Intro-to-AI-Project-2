import random
import math
import numpy as np
import os
from Validator import Validator
from classifier import Classifier

def main():

    print("Welcome to Jeffrey Hwang's Feature Selection Algorithm.\n")
    print("Type in the name of the file to test: ")

    data_input = str(input())
    data_path = os.path.join(os.getcwd(), data_input)
    data = np.genfromtxt(data_path)
    
    print("Type the number of the algorithm you want to run.\n")
    print("\t\t1) Forward Selection")
    print("\t\t2) Backward Elimination")
    alg_input = int(input())
    empty_set = []
    full_set = list(range(1, data.shape[1]))

    print(f'\n\nThis dataset has {data.shape[1]-1} features (not including the class attribute), with {data.shape[0]} instances')
    
    if(alg_input == 1):
        print('\n\nUsing no features and leave one cross validation, I get an accuracy of ' + str(leave_one_cross_validation(data, empty_set)) + '\n')
        forward_selection(data)
    if(alg_input == 2):
        print('\n\nUsing all features and leave one cross validation, I get an accuracy of ' + str(leave_one_cross_validation(data, full_set)) + '\n')
        backward_elimination(data)

def leave_one_cross_validation(data, current_set):
    classifier = Classifier(data)
    validator = Validator(data)
    return validator.accuracy(current_set, classifier)
    
    
    
def forward_selection(data):
    features = data.shape[1]
    current_set_of_features = []
    best_features = []
    best_accuracy = 0
    print('Beginning search\n')
    
    for i in range(1, features):
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, features):
            if k in current_set_of_features:
                continue
            temp_features = current_set_of_features + [k]
            accuracy = leave_one_cross_validation(data, temp_features)
            print(f"\tUsing feature(s) {temp_features} accuracy is {accuracy}")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = k

        if feature_to_add_at_this_level:
            current_set_of_features.append(feature_to_add_at_this_level)
            if best_so_far_accuracy > best_accuracy:
                best_accuracy = best_so_far_accuracy
                best_features = current_set_of_features.copy()
                print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy}\n")
            else:
                print("\n(Warning, Accuracy has decreased!)\n")
    print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy}")
    
def backward_elimination(data):
    features = data.shape[1]
    current_set_of_features = list(range(1, features))
    best_accuracy = 0
    best_features = list(range(1, features))
    print('Beginning search\n')
    
    for i in range(1, features):
        feature_to_remove = []
        best_so_far_accuracy = 0

        for k in range(1, features):
            if k not in current_set_of_features:
                continue
            temp_features = current_set_of_features.copy()  # Create a copy of current_set_of_features
            temp_features.remove(k)  # Remove the element k from test_features
            accuracy = leave_one_cross_validation(data, temp_features)
            print(f"\tUsing feature(s) {temp_features} accuracy is {accuracy}")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove = k

        if feature_to_remove:
            current_set_of_features.remove(feature_to_remove)
            if best_so_far_accuracy > best_accuracy:
                best_accuracy = best_so_far_accuracy
                best_features = current_set_of_features.copy()
                print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy}\n")
            else:
                print("\n(Warning, Accuracy has decreased!)\n")
    print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy}")


if __name__ == "__main__":
    main()