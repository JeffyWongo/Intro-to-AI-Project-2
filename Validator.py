from classifier import Classifier
import numpy as np

class Validator:
    def __init__(self, data):
        self.data = data
    
    def accuracy(self, feature_subset, NNclassifier):
        feature_data = self.data[:, feature_subset]
        number_correctly_classified = 0
        for i in range(len(self.data)):
            test_row = feature_data[i] # store test row
            
            # test_features = test_row[1:] #store all features of row
            test_label = self.data[i][0] #store class of row
            
            training_data = []
            if i == 0:
                training_data = feature_data[1:]
            elif i == (len(self.data)-1):
                training_data = feature_data[:i]
            else:
                training_data = np.concatenate((feature_data[:i],feature_data[i+1:]), axis=0)

            NNclassifier.Train(training_data)
            nn_location, nn_label = NNclassifier.Test(test_row, i)
            # print("Object " + str(i) + " is class " + str(test_label))
            # print("Its nearest neighbor is " + str(nn_location) + " which is in class " + str(nn_label))
            if test_label == nn_label:
                number_correctly_classified = number_correctly_classified + 1

        accuracy = number_correctly_classified / len(self.data)
        return accuracy
       
    
    
    # List of column indices to store
    # feature_indices = current_set.append(feature_to_add)
    # features = data[:, feature_indices]
    
    # number_correctly_classified = 0
    # test_data = self.feature_data[test_data_row_index] # store test row
    #     test_features = test_data[1:] #store all features of row
    #     test_label = test_data[0] #store class of row
    #for i in range(len(features)):
    #     test = features[i] #store first row
    #     test = test[1:] #store all features of row
    #     label_test = test[0] #store class of row
        
    #     print("Looping over i, at the " + str(i) + " location")
    #     print("The " + str(i) + "th object is in class " + str(label_test))
        
    #     nearest_neighbor_distance = float('inf')
    #     nearest_neighbor_location = float('inf')
    #     for k in range(len(features)):
    #         print("Ask if " + str(i) + " is nearest neighbor with " + str(k))
            
    #         if k != i:
    #             distance = math.sqrt(sum((test[k] - features[i][k]) ** 2))
    #             if distance < nearest_neighbor_distance:
    #                 nearest_neighbor_distance = distance
    #                 nearest_neighbor_location = k
    #                 nearest_neighbor_label = features[nearest_neighbor_location][0]