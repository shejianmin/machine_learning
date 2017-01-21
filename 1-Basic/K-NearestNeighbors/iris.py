from sklearn import datasets
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
# save data
# f = open("iris.data.csv", 'wb')
# f.write(str(iris))
# f.close()

print iris

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print "hello"
# print ("predictedLabel is :" + predictedLabel)
print predictedLabel
