import csv
from sklearn.model_selection import train_test_split

X = list()
with open('test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    	X.append(row)

X_train, X_test, y_train, y_test = train_test_split(X, range(len(X)), test_size=0.33, random_state=1232019)

amit = open("newtrain.csv", 'w')
for i in range(len(X_train)):
	amit.write(','.join(X_train[i]) + "\n")
amit.close()

gunjan = open("newtest.csv", 'w')
for i in range(len(X_test)):
	gunjan.write(','.join(X_test[i]) + "\n")
gunjan.close()
