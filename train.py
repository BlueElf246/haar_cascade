from main import *
def train(t):
    with open("training.pkl", 'rb') as f:
        training= pickle.load(f)
    clf= ViolaJones(T=t)
    clf.train(training, 2429, 4548)
    evaluate(clf, training)
    clf.save(str(t))
def test(filename):
    with open("test.pkl", 'rb') as f:
        test= pickle.load(f)
    clf= ViolaJones.load(filename)
    evaluate(clf, test)
def evaluate(clf, data):
    correct =0
    for x,y in data:
        correct+=1 if clf.classify(x)== y else 0
    print(f"Classified {correct} out of {len(data)}")

train(9)
test('9')