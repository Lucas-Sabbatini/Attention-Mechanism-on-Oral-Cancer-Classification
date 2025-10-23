from sklearn import preprocessing

class Normalization:
    def normalize_data(self, X):
        return preprocessing.normalize(X)
    
if __name__ == '__main__': 
    normalizer = Normalization()
    data = [[1, 2, 3], [4, 5, 6]]
    normalized_data = normalizer.normalize_data(data)
    print("Normalized Data:", normalized_data)