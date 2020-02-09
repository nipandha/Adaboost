from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
trn_target, trn_feature = [], []
tst_label, tst_feature = [], []
k = 4
def train_get_T_star():
dt_stump = DecisionTreeClassifier(max_depth=1)
min_error, bestK = float("inf"), 0
X = []
Y = []
std = []
for exp in range(1, k):
print("K = ", str(exp))
clf = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=10**exp)
scr = cross_val_score(clf, trn_feature, trn_target, cv=10)
current_error = 1 - scr.mean()
if current_error < min_error:
min_error = current_error
KStar = exp
err = []
for score in scr:
err.append(1-score)
X.append(exp)
Y.append(current_error)
std.append(scr.std())
draw_cv_k(X, Y, std)
return (10**KStar)
def draw_cv_k(X, Y, std):
plt.figure()
plt.title("Mean CV Error and Std. Deviation over k")
plt.errorbar(X, Y, yerr=std)
plt.xlim(0,k+1)
plt.xlabel("K s.t. T=10^k")
plt.ylabel("Cross Validation Error & Std. Deviation, Folds=10")
plt.show()
def draw_error_per_iteration( T, training_error, test_error ):

plt.figure()
plt.title("Error per iteration")
plt.plot([t for t in range(T)], training_error)
plt.xlim(0, T+1)
plt.xlabel("T till T* from cross validation")
plt.ylabel("Error")
plt.show()
plt.figure()
plt.title("Error per iteration")
plt.plot([t for t in range(T)], test_error)
plt.xlim(0, T+1)
plt.xlabel("T till T* from cross validation")
plt.ylabel("Error")
plt.show()
def test_predict( T ):
dt_stump = DecisionTreeClassifier(max_depth=1)
clfTRAIN = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=T)
clfTRAIN.fit( trn_feature, trn_target )
scrTRAIN = list(clfTRAIN.staged_score(trn_feature, trn_target))
training_error = []
for score in scrTRAIN:
training_error.append(1-score)
print("TrainErrorWithT=", str(T), str(training_error))
yTEST = clfTRAIN.predict( tst_feature )
scrTEST = list(clfTRAIN.staged_score(tst_feature, tst_label))
test_error = []
for score in scrTEST:
test_error.append(1-score)
draw_error_per_iteration(T, training_error, test_error)
def pre_processing():
lbl, ftr = [], []
inpD = "/home/user/Documents/Course Books/FML/HW3/Spambase.txt"
outD = "/home/user/Documents/Course Books/FML/HW3/SpambaseFeatureSet"
lblD = "/home/user/Documents/Course Books/FML/HW3/SpambaseDataLabel"
with open(inpD, 'r') as finpD:
for email in finpD: # For every email in the given data
row = email.strip().split(",") # Separate the columns
lbl.append( row[ len(row) - 1 ] ) # Add label to label vector
ftr.append( row[ :len(row) - 1 ] ) # remove label field from the data vector
with open(outD, 'w+') as foutD:

for fr in ftr:
foutD.write(",".join(fr) + "\n")
with open(lblD, "w+") as flblD:
flblD.write("\n".join(lbl))
def separate_training_test(training, testing, datafn):
trainf = open( training, "w+" )
testf = open( testing, "w+" )
with open( datafn, "r" ) as df:
idx = 0
while( idx < 3450 ):
trainf.write( df.readline() )
idx += 1
testlines = df.readlines()
for tline in testlines:
testf.write(tline)
#print(len(testlines))
testf.close()
trainf.close()
def get_data_ftr( train, test ):
ftrain = open(train, 'r')
for line in ftrain:
feature = list(map(float, line.strip().split(',')))
trn_feature.append(feature)
ftrain.close()
ftest = open(test, 'r')
for line in ftest:
feature = list(map(float, line.strip().split(',')))
tst_feature.append(feature)
ftest.close()
#print(len(trn_feature), len(tst_feature), trn_feature[0])
def get_data_lbl( train, test ):
ftrain = open(train, 'r')
for line in ftrain:
label = int(line.strip())
trn_target.append(label)
ftrain.close()
ftest = open(test, 'r')
for line in ftest:
label = int(line.strip())
tst_label.append(label)

ftest.close()
#print(len(trn_target), len(tst_label), trn_target[4])
def train_test_separate():
trn = "/home/user/Documents/Course Books/FML/HW3/TrainData"
tst = "/home/user/Documents/Course Books/FML/HW3/TestData"
dat = "/home/user/Documents/Course Books/FML/HW3/SpambaseFeatureSet"
separate_training_test(trn, tst, dat)
get_data_ftr(trn, tst)
trn = "/home/user/Documents/Course Books/FML/HW3/TrainDataLabel"
tst = "/home/user/Documents/Course Books/FML/HW3/TestDataLabel"
dat = "/home/user/Documents/Course Books/FML/HW3/SpambaseDataLabel"
separate_training_test(trn, tst, dat)
get_data_lbl(trn, tst)
def main():
pre_processing()
train_test_separate()
T = train_get_T_star()
test_predict(T)
if __name__ == '__main__':
main()
