import math
import numpy as np

class Classifier:
    def __init__(self, data):
        self.training_data = []
        self.data = data
        
    def Train(self, training_set):
        # training_data = []
        # for i in range(len(training_set_row_indices)):
        #     index = training_set_row_indices[i]
        #     training_data = training_data.append(self.feature_data[index])
        self.training_data = training_set
        return
            
    def Test(self, test_row, test_index):
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = -1  # Initialize with a default value
        nearest_neighbor_label = None  # Initialize with a default value
        for index, training_row in enumerate(self.training_data):
            total_distance = 0
            for i in range(len(training_row)):
                distance = (test_row[i] - training_row[i]) ** 2
                total_distance += distance
            final_distance = math.sqrt(total_distance)
            if final_distance < nearest_neighbor_distance:
                nearest_neighbor_distance = final_distance
                if index < test_index:
                    nearest_neighbor_location = index
                else:
                    nearest_neighbor_location = index + 1
                nearest_neighbor_label = self.data[nearest_neighbor_location][0]
        
        return nearest_neighbor_location, nearest_neighbor_label