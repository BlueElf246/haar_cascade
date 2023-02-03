from Cascade import *
import pickle
def train(name):
    with open(name, "rb") as f:
        training= pickle.load(f)
    clf= CascadeClassifier([1,5,10,50])
    clf.train(training)
    evaluate(clf, training)
    clf.save('cascade_1_5_10_50')
def test(name):
    with open(name, 'rb') as f:
        test= pickle.load(f)
    clf= CascadeClassifier.load('cascade_1_5_10_50')
    evaluate(clf, test)
def evaluate(clf, test):
    count=0
    for ex in test:
        if clf.classify(ex[0]) == ex[1]:
            count+=1
    print(f"correct {count} out of {test.shape[0]}")

train('car_train.pkl')
test('car_test.pkl')


