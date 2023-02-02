from main import *
import pickle
class CascadeClassifier():
    def __init__(self, layers):
        self.layers = layers
        self.clfs=[]
    def train(self, training):
        pos, neg=[], []
        for ex in training:
            if ex[1]==1:
                pos.append(ex)
            else:
                neg.append(ex)
        for feature_num in self.layers:
            if len(neg)==0:
                print('Stopping early, since FPR =0')
                break
            clf= ViolaJones(T=feature_num)
            clf.train(pos+neg, len(pos), len(neg))
            self.clfs.append(clf)
            false_positive=[]
            for ex in neg:
                if clf.classify(ex[0])== 1:
                    false_positive.append(ex)
            neg=false_positive
    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image)==0:
                return 0
        return 1
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)



