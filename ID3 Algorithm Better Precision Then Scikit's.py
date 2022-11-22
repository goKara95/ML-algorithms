############################################   PART 1   ######################################################################
import numpy as np
import pandas as pd
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv") # DIKKATTTTTTTT unicode_escape
df = shuffle(df)
dataset = df.to_numpy()
#IN HERE WE SWITCHED ATTRITION AND AGE COLUMNS SINCE IT WAS HARDER TO MANIPULATE THE TABLE
foo = df.to_numpy()
for i in range(len(foo)):
    dataset[i][0] = foo[i][1]
    dataset[i][1] = foo[i][0]
Main_Attributes = list(df.columns)
Main_Attributes[0] = "Attrition"
Main_Attributes[1] = "Age"

#DISCRITIZATION:
#FIND MAX AND MINIMUM OF EACH COLUMN. IF THEY DIFFER BY MORE THAN 8 DIVIDE THEM INTO 9 SUB-GROUPS
#STARTING POINT OF EACH GROUP IS LISTED IN TEMP. IF AN ATTRIBUTES VALUE IS SMALLER OR EQUAL TO AN INDEX IN TEMP THAT INDEX WILL
#ASSIGNED IT AS NEW ATTRIBUTE
#EmployeeCount, StandardHours are redundant
for j in range(len(Main_Attributes)):
    if isinstance(dataset[0][j], int):
        temp = []
        min = df[Main_Attributes[j]].min() #since it will be used below storing it here is a better way
        distance = df[Main_Attributes[j]].max() - min
        if distance > 8:
            temp.append(min)
            gap = distance/8
            for i in range(1,9):
                temp.append(min+gap*i)
            for i in range(len(dataset)):
                for index in temp:
                    if dataset[i][j] <= index:
                        att_name = Main_Attributes[j]
                        if temp.index(index) !=0:
                            dataset[i][j] = str(temp[temp.index(index)-1]) + "<" + att_name + "<=" + str(index)
                        else:
                            dataset[i][j] = att_name + "=" + str(index)
                        break

df = pd.DataFrame(dataset, columns = Main_Attributes)

sub_attr = []
for i in range(len(Main_Attributes)):
    uniq = np.unique(dataset[:, i])
    for j in range(len(uniq)):
        sub_attr.append(uniq[j])

Main_Attributes.pop(0)
main_attr = set(Main_Attributes)

print(df)


test1 = np.copy(dataset[:294])
df1 =  df.iloc[294:]
train1 = df1.to_numpy()
y_true1 = np.copy(dataset[:294,0])
test1 = np.delete(test1, 0, 1)


test2 = np.copy(dataset[294:588])
df2 = pd.concat([df.iloc[0:294], df.iloc[588:]], axis=0)
train2 = df2.to_numpy()
y_true2 = np.copy(dataset[294:588,0])
test2 = np.delete(test2, 0, 1)



test3 = np.copy(dataset[588:882])
df3 = pd.concat([df.iloc[0:588], df.iloc[882:]], axis=0)
train3 = df3.to_numpy()
y_true3 = np.copy(dataset[588:882,0])
test3 = np.delete(test3, 0, 1)



test4 = np.copy(dataset[882:1176])
df4 = pd.concat([df.iloc[0:882], df.iloc[1176:]], axis=0)
train4 = df4.to_numpy()
y_true4 = np.copy(dataset[882:1176,0])
test4 = np.delete(test4, 0, 1)


test5 = np.copy(dataset[1176:])
df5 = df.iloc[:1176]
train5 = df5.to_numpy()
y_true5 = np.copy(dataset[1176:,0])
test5 = np.delete(test5, 0, 1)


# this function prints the pruning rules
def print_rules(root, main_attr, sub_attr):
    # array to store the lines
    rules_arr = []

    # recursive function to add the attributes and required characters
    def rules(root, main_attr, sub_attr, string):
        # base case of the recursion
        if len(root.children) == 0:
            if root.name == "True":
                string += " Then Attrition=Yes"
            elif root.name == "False":
                string += " Then Attrition=No"
            # apeend the line to the array
            rules_arr.append(string)
            return
        # if node is a main attribute
        if root.name in main_attr:
            string += "("
            string += str(root.name)
            string += "="

        # if node is a sub attribute
        elif root.name in sub_attr:
            string += str(root.name)
            if len(root.children[0].children) == 0:
                string += ")"
            else:
                string += ") ∧ "

        # recursion
        for child in root.children:
            rules(child, main_attr, sub_attr, string)

    string = "if "  # all lines start with "if"
    # call the recursive function to append the lines to the rules_arr
    rules(root, Main_Attributes, sub_attr, string)
    # print every line in the rules_arr properly
    for i in range(len(rules_arr)):
        temp = rules_arr[i]
        string = "R"
        string += str(i + 1)
        string += ": "
        string += temp
        rules_arr[i] = string
        print(rules_arr[i])


class Node:
  def __init__(self, name):
    self.name = name
    self.parent = None
    self.children = []
    self.count = None
  def add_attr(self, name):
    o1 = Node(name)
    self.children.append(o1)
    o1.parent = self
  def add_back_node(self, Node):
    Node.parent = self
    self.children.append(Node)
  def delete(self, name):
    for i in range(len(self.children)):
        if self.children[i].name == name:
            self.children.pop(i)
  def remove_self(self):
    label = self.children[0].children[0].name
    flag = 1
    for i in range(len(self.children)):
        for j in range(len(self.children[i].children)):
            if self.children[i].children[j].name != label:
                flag = 0
                break
    if flag == 1:
        self.name = label
        self.children = []

def entropy(posValue, negValue):
    if posValue == 0 or negValue == 0:
        return 0
    else:
        total = posValue + negValue
        return -((posValue/total) * math.log((posValue/total),2)) - ((negValue/total) * math.log((negValue/total),2))

# information gain function
# argument arr is a numpy 2d array with 2 index in each array
def infoGain(arr):
    totalArray = np.sum(arr, 0) # sum of the columns of the array
    totalNum = np.sum(totalArray) # sum of every element in the array
    # get the total entropy for calculations
    totalEntropy = entropy(totalArray[0], totalArray[1])
    gain = totalEntropy
    for i in arr:
        entropy_for_label = entropy(i[0], i[1]) # calculate the entropy for each
        total_for_label = np.sum(i)
        gain -= (total_for_label/totalNum) * entropy_for_label
    return gain


# this function returns the attribute with the maximum information gain
def Gains(Samples, Attr):
    # get the uniq values in the column
    arrays_of_uniq = []
    for i in range(len(Samples[0, 1:])):  # önceki Samples[0, 1:] idi
        uniq = np.unique(Samples[:, i + 1])  # önceki i yerine i+1 idi
        arrays_of_uniq.append(uniq)

    store_infoGain = []  # it is an 2D array, it stores the attribute and its information gain
    for i in range(0, len(arrays_of_uniq)):
        parent_pos_neg = []
        # traverse different attributes from same type
        for j in range(len(arrays_of_uniq[i])):
            pos = 0
            neg = 0
            # traverse the data
            for k in range(len(Samples)):
                if (arrays_of_uniq[i][j] in Samples[k]):
                    if (Samples[k][0] == "Yes"):
                        pos += 1
                    elif (Samples[k][0] == "No"):
                        neg += 1
            parent_pos_neg.append([pos, neg])

        # find info gain
        infoGain_for_parent = infoGain(parent_pos_neg)

        # append the attribute and its information gain to store_infoGain
        lol = []
        if Main_Attributes[i] in Attr:
            lol.append(infoGain_for_parent)
            lol.append(Main_Attributes[i])
            store_infoGain.append(lol)

    # sort the array to find the maximum information gain
    store_infoGain = sorted(store_infoGain)

    if len(store_infoGain) == 0:
        return "Error, empty"
    else:
        return store_infoGain[len(store_infoGain) - 1][1]  # returns the attribute with the maximum information gain


def ID3(df, Attributes, Sub_Attributes, root):
    # tüm examplelar yes'se bakmaya gerek yok yes nodu ekle geç
    Space = np.copy(df.to_numpy())
    if "No" not in set(Space[:, 0]):
        if root.name in Attributes:
            Attributes.remove(root.name)
        elif root.name in Sub_Attributes:
            Sub_Attributes.remove(root.name)
        root.add_attr("True")
        root.children[len(root.children) - 1].count = len(Space[:, 0])
        return
    # tüm examplelar no ise bakmaya gerek yok yes nodu ekle geç
    elif "Yes" not in set(Space[:, 0]):
        if root.name in Attributes:
            Attributes.remove(root.name)
        elif root.name in Sub_Attributes:
            Sub_Attributes.remove(root.name)
        root.add_attr("False")
        root.children[len(root.children) - 1].count = len(Space[:, 0])
        return
    elif len(Attributes) == 0:
        if np.count_nonzero(Space[:, 0] == "Yes") > np.count_nonzero(Space[:, 0] == "No"):
            root.add_attr("True")
            root.children[len(root.children) - 1].count = np.count_nonzero(
                Space[:, 0] == "Yes")  # eşittirin ağ tarafı buydu: len(Space[:, 0])
        else:
            root.add_attr("False")
            root.children[len(root.children) - 1].count = np.count_nonzero(Space[:, 0] == "No")  # len(Space[:, 0])
        return
        # return root

    elif root.name in Attributes:
        # attributestan attr adını sil, childrenlarını dataframeden bul, ve leaf olarak ekle
        # her leaf için fonksiyonu çağır, tabii ki leaf'in subspace ile.
        Attributes.remove(root.name)
        children = df[root.name].unique()
        for i in range(len(children)):
            root.add_attr(children[i])
        for child in root.children:
            ID3(df[(df == child.name).any(axis=1)], Attributes, Sub_Attributes, child)
        # return root

    else:  # elif root.name in Sub_Attributes, accuracy düşük falan çıkarsa bunu bir de else çevirip dene. yani sub_Attr'da olma koşuluna bakma. zaten sub_Attr olmayan bir şey denk gelmez herhalde
        # attrlardan sil, gainsi en yüksek olan attr'i child olarak ekle
        # aynı spacete child'ı kullanarak fonksiyonu çağır
        # Sub_Attributes.remove(root.name)
        root.add_attr(Gains(Space[1:, :], Attributes))  # returns the element with maximum gain
        ID3(df, Attributes, Sub_Attributes, root.children[0])
        # return root

# space'i df olarak yollayıp işlem yapacağım zaman numpy'a çevirmeliyim.

#buna yeni bir şey ekledim none returnlemesin diye hiçbir şey bulamazsa "No" returnleyecek
def Predict(test, root):
    if root.name == "True":
        return "Yes"
    elif root.name ==  "False":
        return "No"
    elif root.name in Main_Attributes:
        for child in (root.children):
            if child.name in test:
                return Predict(test,child)
                break
    elif root.name in test:
        return Predict(test,root.children[0])#root.name test edilen durumun içindeyse main değil sub attributetur ve sub attrlarda sadece 1 çocuk olur
    elif root.name is None:
        return "No"


def calculate_accuracy(root, test, y_true):
    suc = 0
    fail = 0
    for i in range(len(test)):
        # predict the label
        str1 = Predict(test[i], root)
        str2 = y_true[i]
        # check if the predicted label and true labelis same
        if (str1 == str2):
            suc += 1
        else:

            fail += 1

    accuracy = suc / (suc + fail)
    return accuracy


# FOLD1
root1 = Node(Gains(train1, main_attr))
ID3(df1, main_attr, sub_attr, root1)
prediction1 = []
for i in range(294):
    a = Predict(test1[i], root1)
    if a is None:
        prediction1.append("Yes")
    else:
        prediction1.append(a)

# FOLD2 main and sub attr are created again since they are altered during operations
Main_Attributes = list(df.columns)
sub_attr = []
for i in range(len(Main_Attributes)):
    uniq = np.unique(dataset[:, i])
    for j in range(len(uniq)):
        sub_attr.append(uniq[j])
Main_Attributes.pop(0)
main_attr = set(Main_Attributes)

root2 = Node(Gains(train2, main_attr))
ID3(df2, main_attr, sub_attr, root2)
prediction2 = []
for i in range(294):
    a = Predict(test2[i], root2)
    if a is None:
        prediction2.append("Yes")
    else:
        prediction2.append(a)

# FOLD3
Main_Attributes = list(df.columns)
sub_attr = []
for i in range(len(Main_Attributes)):
    uniq = np.unique(dataset[:, i])
    for j in range(len(uniq)):
        sub_attr.append(uniq[j])
Main_Attributes.pop(0)
main_attr = set(Main_Attributes)

root3 = Node(Gains(train3, main_attr))
ID3(df3, main_attr, sub_attr, root3)
prediction3 = []
for i in range(294):
    a = Predict(test3[i], root3)
    if a is None:
        prediction3.append("Yes")
    else:
        prediction3.append(a)

# FOLD4
Main_Attributes = list(df.columns)
sub_attr = []
for i in range(len(Main_Attributes)):
    uniq = np.unique(dataset[:, i])
    for j in range(len(uniq)):
        sub_attr.append(uniq[j])
Main_Attributes.pop(0)
main_attr = set(Main_Attributes)

root4 = Node(Gains(train4, main_attr))
ID3(df4, main_attr, sub_attr, root4)
prediction4 = []
for i in range(294):
    a = Predict(test4[i], root4)
    if a is None:
        prediction4.append("Yes")
    else:
        prediction4.append(a)

# FOLD5
Main_Attributes = list(df.columns)
sub_attr = []
for i in range(len(Main_Attributes)):
    uniq = np.unique(dataset[:, i])
    for j in range(len(uniq)):
        sub_attr.append(uniq[j])
Main_Attributes.pop(0)
main_attr = set(Main_Attributes)

root5 = Node(Gains(train5, main_attr))
ID3(df5, main_attr, sub_attr, root5)
prediction5 = []
for i in range(294):
    a = Predict(test5[i], root5)
    if a is None:
        prediction5.append("Yes")
    else:
        prediction5.append(a)

from sklearn.metrics import classification_report
# Classification Performance Metric for fold1
print(classification_report(y_true1, prediction1))

# Classification Performance Metric for fold2
print(classification_report(y_true2, prediction2))

# Classification Performance Metric for fold3
print(classification_report(y_true3, prediction3))

# Classification Performance Metric for fold4
print(classification_report(y_true4, prediction4))

# Classification Performance Metric for fold5
print(classification_report(y_true5, prediction5))

# Finding the best accuracy between 5 fold and writing its rules

Main_Attributes = list(df.columns)
# calculate the accuracy for every fold and find the fold with the highest  accuracy
accuracy_folds = []
accuracy_folds.append(calculate_accuracy(root1, test1, y_true1))
accuracy_folds.append(calculate_accuracy(root2, test2, y_true2))
accuracy_folds.append(calculate_accuracy(root3, test3, y_true3))
accuracy_folds.append(calculate_accuracy(root4, test4, y_true4))
accuracy_folds.append(calculate_accuracy(root5, test5, y_true5))
# find the index that has highest accuracy value
max_accuracy = max(accuracy_folds)
index_max_accuracy = accuracy_folds.index(max_accuracy)

if index_max_accuracy == 0:
    print("Fold1 has the highest accuracy")
    print_rules(root1, Main_Attributes, sub_attr)
elif index_max_accuracy == 1:
    print("Fold2 has the highest accuracy")
    print_rules(root2, Main_Attributes, sub_attr)
elif index_max_accuracy == 2:
    print("Fold3 has the highest accuracy")
    print_rules(root3, Main_Attributes, sub_attr)
elif index_max_accuracy == 3:
    print("Fold4 has the highest accuracy")
    print_rules(root4, Main_Attributes, sub_attr)
elif index_max_accuracy == 4:
    print("Fold5 has the highest accuracy")
    print_rules(root5, Main_Attributes, sub_attr)

############################################   PART 2   ######################################################################
df = shuffle(df)
dataset = df.to_numpy()
prune_test = np.copy(dataset[:294]) # test set to calculate accuracy before pruning and after pruning
validation = np.copy(dataset[294:598]) # validation set to use in pruning process
df1 =  df.iloc[598:] # train set dataframe
train = df1.to_numpy() # convert the train set to numpy array
test_true = np.copy(dataset[:294,0]) # true labels of the test set
validation_true = np.copy(dataset[294:598,0]) # true labels of the validation set

# creating the tree
Main_Attributes = list(df.columns)
sub_attr = []
for i in range(len(Main_Attributes)):
    uniq = np.unique(dataset[:, i])
    for j in range(len(uniq)):
        sub_attr.append(uniq[j])
Main_Attributes.pop(0)
main_attr = set(Main_Attributes)

root = Node(Gains(train, main_attr))
ID3(df1, main_attr, sub_attr, root)


# twigs is a 2d array twig[i][0] is twig, twig[i][1] parent of twig
def findTwigs(root, twigs):
    temp = []  # temp'i twigs'e 2 elemanlı array eklemek için kullandım
    Flag = 1  # 1 means twig 0 means not twig
    if root.name in Main_Attributes:
        for child in root.children:  # torunlarının(çocuğunun çocuğu) hepsi yes veya no olan nodelar twigtir
            # sadece Main_Attr olan bir node twig olabilir. Her Main_Attr nodeunun tüm çocuklarını dolaşıyoruz, torunlarından birinin
            # bile değeri yes veya no değilse loopu kırıyoruz ve ağaçta aşağı inmek için sırasıyla çocuklarıyla fonksiyonu çağırıyoruz
            if child.children[0].name != "True" and child.children[0].name != "False":
                Flag = 0
                break
        if Flag == 1:  # eğer tüm torunlar True/False ise flag hiçbir zaman 0 olmaz ve aynen kalır.
            # twigs'e node'un kendisini ve annesini ekliyoruz. annenin çocuklarından node'u çıkarıyoruz.
            # eğer accuracy düşerse twig'imizi geri eklemek için tek yapmamız gereken parent.add_back_node() fonksiyonunu kullanmak
            # bu sayede silinen node'dan sonraki subtree'yi kaybetmemiş olacağız.
            temp.append(root)
            temp.append(root.parent)
            twigs.append(temp)
            # root.parent.delete(root.name) silme burada yapılmayacak, bu fonksiyonu sadece twig bulmak için kullan

    for child in root.children:
        findTwigs(child, twigs)


def findTwigInfoGain(twigs):
    # infoGains[i][0] is information gain of twig, infoGains[i][1] is name of the twig
    infoGains = []

    # loop for main attributes (twigs)
    for twig in twigs:
        # 2d array to store the number of yes and no of twig (arguament of infoGain function)
        parent_yes_no = []
        # loop for sub attributes of the twig
        for i in range(len(twig[0].children)):
            # if grandchildren of twig is yes
            if (twig[0].children[i].children[0].name == "True"):
                # first index is the number of "Yes", second index is 0 since there isn't a "No" in "Yes" leaf
                tempArr = [twig[0].children[i].children[0].count, 0]
                parent_yes_no.append(tempArr)

            # if grandchildren of the twig is no
            elif (twig[0].children[i].children[0].name == "False"):
                # first index is 0 since there isn't a "Yes" in "No" leaf, second index is the number of "No"
                tempArr = [0, twig[0].children[i].children[0].count]
                parent_yes_no.append(tempArr)

        # temp array to store info gains and corresponding twig ( [information gain of twig, name of the twig] )
        temp = []
        temp.append(infoGain(parent_yes_no))
        temp.append(twig[0].name)
        infoGains.append(temp)

    # sort the twigs to find the twig with the least information gain
    infoGains = sorted(infoGains)
    least_info_gain_node_name = infoGains[0][1]  # get the name of twig with the least information gain

    # get the node of twig with the least information gain using the name of the twig
    for twig in twigs:
        if twig[0].name == least_info_gain_node_name:
            twig_node_del = twig[0]

    return twig_node_del  # return the node of the twig with the least information gain


def deleteTwig(twig_node_del):
    # find the number of yes and no to find the label of leaf
    yes_count = 0
    no_count = 0
    for i in range(len(twig_node_del.children)):
        if twig_node_del.children[i].children[0].name == "True":
            yes_count += twig_node_del.children[i].children[0].count
        elif twig_node_del.children[i].children[0].name == "False":
            no_count += twig_node_del.children[i].children[0].count

    # get the total count to find the new count of the label of new leaf
    total_count = yes_count + no_count

    # determine label of new leaf
    if (yes_count > no_count):
        label = "True"
    elif (no_count > yes_count):
        label = "False"
    else:  # if count of yes and no is same, we make the label false
        label = "False"

    # store the deleted twig node to revert changes in the future
    deleted_node = twig_node_del
    parentOf_twig = twig_node_del.parent  # get the parent of node

    # if we come to the root of the tree we can't delete the root
    if parentOf_twig == None:
        deleted_node.remove_self()  # rename the node (root) with the label of its grandchildren
        return 0

    # delete the twig
    parentOf_twig.delete(twig_node_del.name)
    # delete from main_attr
    Main_Attributes.remove(twig_node_del.name)

    # after we deleted the twig, add a leaf to the parent of twig
    # name of this leaf is label of the deleted twig ("True" or "False")
    parentOf_twig.add_attr(label)
    parentOf_twig.children[0].count = yes_count + no_count  # count of our newly added leaf

    return deleted_node


# this function returns two value in tuple. First one is the node of deleted twig, second is accuracy
def prune_tree(root, test, y_true):
    # twigs is a 2d array twig[i][0] is twig, twig[i][1] parent of twig
    twigs = []
    # call this function to append the twigs to the twigs array
    findTwigs(root, twigs)

    # call this function to get the node of the twig with the least information gain
    del_node = findTwigInfoGain(twigs)

    # delete the twig with least information gain and store the deleted twig in the deleted_node variable
    deleted_node = deleteTwig(del_node)

    # if deleteTwig function returns 0, this means we reached the root
    if deleted_node == 0:
        return 0, 0

    accuracy = calculate_accuracy(root, test, y_true)

    return deleted_node, accuracy

# before pruning, calculate the accuracy on test set
before_pruning_acc = calculate_accuracy(root, prune_test, test_true)
print("accuracy on test set before : ", before_pruning_acc)

# before pruning print the rules
print_rules(root, main_attr, sub_attr)

# PRUNING PROCESS

# Last Accuracy variable is the accuracy of our decision tree model on validation set before pruning process.
last_accuracy = calculate_accuracy(root, validation, validation_true)

# this loop continues until accuracy drops or we reaches the root
i = 0
while True:
    # call the prune_tre function to get the node of the deleted twig and accuracy
    prune_tree_tuple = prune_tree(root, validation, validation_true)
    deleted_node = prune_tree_tuple[0]  # node of the deleted twig
    current_accuracy = prune_tree_tuple[1]  # current accuracy with removed twig

    if deleted_node == 0:  # this means we reached the root
        break

    # this means accuracy dropped after we deleted the twig so top pruning and revert the last changes
    if (current_accuracy < last_accuracy):
        Main_Attributes.append(deleted_node.name)  # add back the name of twig to the main attributes array
        parentOf_deleted_node = deleted_node.parent  # get the parent of the deleted twig
        delete_label = parentOf_deleted_node.children[0].name  # get the name of deleted twig
        parentOf_deleted_node.delete(delete_label)  # delete the node we added to the tree after deleting twig
        parentOf_deleted_node.add_back_node(deleted_node)  # finally, add back the twig to the tree
        break
    i += 1
    last_accuracy = current_accuracy


# after pruning, calculate the accuracy on test set
after_pruning_acc = calculate_accuracy(root, prune_test, test_true)
print("after_pruning_acc: ", after_pruning_acc)

# after pruning print the rules
print_rules(root, main_attr, sub_attr)



