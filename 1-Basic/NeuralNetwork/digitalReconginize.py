# from sklearn.datasets import load_digits
# 
# digits= load_digits()
# print(digits.data.shape)
# 
# import pylab as pl
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()


from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report

digits = load_digits()
x= digits.data
y = digits.target
x -= x.min()
x /= x.max()

nn = NeuralNetwork([64,100,10],"logistic")
x_train,x_test,y_train,y_test = train_test_split(x,y)
label_train = LabelBinarizer().fit_transform(y_train)
label_test = LabelBinarizer().fit_transform(y_test)
print("start fitting..")
predictions = []
nn.fit(x_train, label_train, epochs=10000)
for i in range(x_test.shape[0]):
    o = nn.predict(x_test[i])
    predictions.append(np.argmin(o))
print confusion_matrix(y_test,predictions)
print classification_report(y_test,predictions)