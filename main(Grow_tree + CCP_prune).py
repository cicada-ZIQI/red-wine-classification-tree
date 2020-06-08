"""divide the data into left branch or right branch according to the split number"""
def split_data_set_left(data_set, axis, mean_value):
    # All examples which is smaller than or equal to the best mean value under the best feature are divided into the left branch
    # All the data in the left brach have a smaller value of the feature than the split number. 
    new_data_set_left = []
    
    for row in data_set:
        if row[axis] <= mean_value:
            split_vector = row[:axis]
            split_vector.extend(row[axis + 1:])
            new_data_set_left.append(split_vector)
    
    return new_data_set_left

def split_data_set_right(data_set, axis, mean_value):
    # All examples which is greater than the best mean value under the best feature are divided into the right branch
    # All the examples in the roght brach have a bigger value of the feature than the split number.
    new_data_set_right = []
    
    for row in data_set:
        if row[axis] > mean_value:
            split_vector = row[:axis]
            split_vector.extend(row[axis + 1:])
            new_data_set_right.append(split_vector)

    return new_data_set_right

"""calculate gini of the data"""
def calc_gini(data_set):

    count = len(data_set)
    label_counts = {}

    # For all examples in each branch: calculate the number of examples that have the lable ">6" and the number of examples that have the lable "<=6"
    # Store the number of data whose lable is ">6" and the number of data whose label is"<=6" into a dictionary.
    for row in data_set:
        label = row[-1]  
        # The lable of each example is the element of the list
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    # Calcilate the gini_index.
    gini = 1.0
    for key in label_counts:
        prob = label_counts[key] / count
        gini -= prob * prob
    return gini

''' Calculate the mean value of two neighbouring feature values as the candidates of split points.'''
def cal_mean(i, j):
    mean = (i+j)/2
    return mean 

'''Find the best split feature according to gini index of each feature.'''
'''Find the best split point under the best split feature.'''
def choose_best_feature_and_split_gini(data_set):

    feature_count = len(data_set[0]) - 1 
    
    # Store the minimum gini among all the feature.
    min_gini_index = 0.0
    # Store in the best feature according to when we get the minimum gini index.
    best_feature = -1

    # Calculate gini index of each feature.
    for i in range(feature_count):
        features = [example[i] for example in data_set]  
        # Get all the values of the examples under that feature.

        feature_value_set = list(set(features))
        means = []      
        # Create a list to store n-1 mean values.
         
        for j in range(len(feature_value_set)-1):
                mean = cal_mean(feature_value_set[j], feature_value_set[j+1])
                means.append(mean)          

        # Calculate gini index of each split number 
        for mean_value in means:

            sub_data_set_left = split_data_set_left(data_set, i, mean_value)  
            sub_data_set_right = split_data_set_right(data_set, i, mean_value)
            prob_left = len(sub_data_set_left) / len(data_set)
            prob_right = len(sub_data_set_right) / len(data_set)
            gini_index = prob_left * calc_gini(sub_data_set_left) + prob_right * calc_gini(sub_data_set_right)

            # Evert time after calculating the gini index of a split point under a specific feature, update the minimum gini index, the best split and the best feature for now.
            if gini_index < min_gini_index or min_gini_index == 0.0:
                min_gini_index = gini_index
                best_feature = i
                best_split = mean_value
     
    return best_feature,best_split

# Find the majority label of all examples in a node.
def get_top_class(class_labels):
    count_class_below = 0
    count_class_above = 0
    for i in class_labels:
        if i == '<=6':
            count_class_below += 1
        else:
            count_class_above += 1
    # If most of the data have the label "<=6", then return "<=6" as the top class, vice versa. 
    # Return the number of the minority data whose label don't equal to the top class
    if count_class_below > count_class_above:
        top_class = '<=6'
        return top_class, count_class_above
    else:
        top_class = '>6'
        return top_class, count_class_below

''' The process of growing the dicision tree.'''
def grow_tree(data_set):
    class_list = []
    # A list to store the labels of all examples in the curent node.

    for example in data_set:
        if example[-1] <= 6:
            class_list.append('<=6')
        else:
            class_list.append('>6')
    
    '''determine whether we should continue to grow the tree.'''
    # If all the features have been considered and split the data according to the best split of that feature,
    # then return the label of the majority 
    if len(data_set[0]) == 1 :
        # The second return value is the number of the minority whose label is different from the majority.
        return get_top_class(class_list)[0], get_top_class(class_list)[1]

    # If all the data in that node have the same label, then return that label and 0 since the number of the minority is 0.
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0],0
     
    '''If we haven't reached the ending condition'''
    # Find the best feature and best split number to divide the data into two parts.
    best_feature,best_split = choose_best_feature_and_split_gini(data_set)
    
    # Use a dictionary to store the tree, including the root and each node.

    tree = {}

    tree['best_feature'] = best_feature
    tree['best_split'] = best_split
    
    # Use recurtion to grow a tree.
    left_set = split_data_set_left(data_set, best_feature, best_split)
     
    right_set = split_data_set_right(data_set, best_feature, best_split)

    tree["amount_<=6"] =  class_list.count("<=6")
    tree["amount_>6"] = class_list.count(">6")
    
    # Every subtree is a value under the key 'left'/'righty' in the original tree.
    if len(left_set) != 0:
        tree['left'] = grow_tree(left_set)
    if len(right_set) != 0:
        tree['right'] = grow_tree(right_set)
     
    return tree    

# Use the form [[],[]……,[]] to store each piece of data in the dataset. Each small list in the large list store one piece of data(an example).
def load_data(file):
    whole_data=[]
    fr = open(file,"r")
    whole = fr.readlines()[1:]
    for line in whole:
        line = line.split('\n')[0].split(", ")
        float_line = []
        for i in line:
            float_line.append(eval(i))
        whole_data.append(float_line)
    fr.close()
    return whole_data
 
# Use the tree to predict the labels of all examples in the test dataset, return that label.
def classify(mytree, test_data):
    test_data_temp=eval(str(test_data))
    if not isinstance(mytree, dict):
        class_label = mytree[0]
        return class_label
    else:
        best_feature = mytree['best_feature'] 
        best_split = mytree['best_split']  
        if test_data_temp[best_feature] <= best_split:
            next_tree = mytree['left']
        elif test_data_temp[best_feature] > best_split:
            next_tree = mytree['right']
        del test_data_temp[best_feature]
        return classify(next_tree, test_data_temp)

# Use the tree to predict the label of each piece of data in the test dataset,
# then compare the predict label with the real label, then get the accuracy of the prediction of the whole data
def calc_accuracy(mytree,test_data_set):
    classification_result = []
    for test_data in test_data_set:
        prediction = classify(mytree, test_data)
        if (test_data[-1] <= 6 and prediction == "<=6") or (test_data[-1] > 6 and prediction == ">6"):
            classification_result.append(True)
        else:
            classification_result.append(False)
    accuracy = classification_result.count(True)/len(classification_result)
    number_of_false=classification_result.count(False)
     
    return accuracy,number_of_false

# calculate error cost of the leaf nodes under a node which isn't a leaf node
# error cost = the number of the minority whose label is different from the predict label(here I don't let it divide the number
# of the whole data since all the error cost need to divide that number, then it is meaningless when caomparing these error cost)
def calc_sub_tree_error_cost(mytree,sub_tree_error_cost):
    if isinstance(mytree["left"],dict):
        sub_tree_error_cost=calc_sub_tree_error_cost(mytree["left"],sub_tree_error_cost)
    else:
        sub_tree_error_cost += mytree["left"][1]
    if isinstance(mytree["right"],dict):
        sub_tree_error_cost=calc_sub_tree_error_cost(mytree["right"],sub_tree_error_cost)
    else:
        sub_tree_error_cost += mytree["right"][1]
    return sub_tree_error_cost

# find how many leaf nodes of a node that isn't a leaf node
def calc_leaf(mytree,leaf):
    if isinstance(mytree,tuple):
        return 0
    if isinstance(mytree["left"],dict):
        leaf=calc_leaf(mytree["left"],leaf)
    else:
        leaf += 1
    if isinstance(mytree["right"],dict):
        leaf=calc_leaf(mytree["right"],leaf)
    else:
        leaf += 1
    return leaf

# calculate alpha of each node that isn't a leaf node
# put (alpha, the subtree whose root node is this node, all the data that is divided into this subtree) as a tuple into a list
def calc_alpha(mytree,data_set,alpha_store):
    if not isinstance(mytree,tuple):
        # R is the error cost of that node which isn't a leaf node
        if mytree["amount_<=6"] <= mytree["amount_>6"]:
            R=mytree["amount_<=6"]
        else:
            R=mytree["amount_>6"]
        sub_tree_error_cost=calc_sub_tree_error_cost(mytree,0) 
        T=calc_leaf(mytree,0)
        alpha = (R - sub_tree_error_cost) / (T - 1)
        alpha_store.append((alpha,mytree,data_set))
     
    best_feature = mytree["best_feature"]
    best_split = mytree["best_split"]
    data_set_left = split_data_set_left(data_set,best_feature,best_split)
    data_set_right = split_data_set_right(data_set,best_feature,best_split)
    if isinstance(mytree["left"],dict):
        calc_alpha(mytree["left"],data_set_left,alpha_store)
    if isinstance(mytree["right"],dict):
        calc_alpha(mytree["right"],data_set_right,alpha_store)
    return alpha_store

# change a suntree whose alpha is the smallest into a leaf node
# use the majority label to represent the predict label of these data 
def change_sub_tree(mytree,sub_tree,new_tree):
    if mytree["left"]==sub_tree:
        mytree["left"]=new_tree
        return mytree
    elif mytree["right"]==sub_tree:
        mytree["right"]=new_tree
        return mytree
    elif mytree == sub_tree:
        mytree["left"]=("<=6",0)
        mytree["right"]=("<=6",0)
        return mytree
    else:
        if isinstance(mytree["left"],dict):
            change_sub_tree(mytree["left"],sub_tree,new_tree)
        if isinstance(mytree["right"],dict):
            change_sub_tree(mytree["right"],sub_tree,new_tree)
        return mytree
 
# use cost-complexity pruning(CCP) to prune the tree
def CCP_prune(mytree,test_data_set,train_data_set):
    # before pruning, find the accuracy of prediction
    accuracy_before=calc_accuracy(mytree,test_data_set)
    # each time after pruning, calculate the accuracy to update accuracy-now
    accuracy_now=accuracy_before
    # store the newly-pruned-tree
    mytree_update=[]
    # put(accuracy of the newly-pruned tree, newly-pruned tree) as a tuple in the list
    accuracy_tree=[]
    # each time after pruning, put the accuracy of predicting the second half of the train dataset into the list
    accuracy=[]
    mytree_update=mytree
    mytree_store=eval(str(mytree))
    accuracy_tree.append((accuracy_now[0],mytree_store))
    # continuously to prune the tree until there is only one node left 
    while calc_leaf(mytree_update,0) > 2 :
        # calculate (alpha, the subtree whose root node is this node, all the data that is divided into this subtree) of each node
        # that isn't a leaf node of the tree
        alpha_store=calc_alpha(mytree_update,train_data_set,[])
        # store all the alpha of each node that isn't a leaf node
        alpha=[]
        # if one node has the minimum alpha, then store 
        # (alpha, the subtree whose root node is this node, all the data that is divided into this subtree) into candidate
        candidate=[]
        # candidate2 store the number of data if the data divided into a node who has the minimum alpha
        candidate2=[]
        accuracy_before=accuracy_now
         
        for i in alpha_store:
            alpha.append(i[0])
        alpha.sort()
        # Each time after getting a newly-pruned tree, find the minimum alpha
        min_alpha=alpha[0]
         
        for i in alpha_store:
            if i[0] == min_alpha:
                candidate.append(i)
                candidate2.append(len(i[2]))
        candidate2.sort(reverse=True)
        # if more than one node has the minimum alpha, choose the node who has the most data to prune
        cut_data_length=candidate2[0]
         
        for i in candidate:
            if len(i[2]) == cut_data_length:
                tree_cut=i[1]
                check_data=i[2]
        # choose label:"<=6" or ">6" when collapse the sub tree     
        error_delete_1 = 1 - calc_accuracy(("<=6",0),check_data)[0]
        error_delete_2 = 1 - calc_accuracy((">6",0),check_data)[0]
        # this is the number of the minority class
        wrong_1=calc_accuracy(("<=6",0),check_data)[1]
        wrong_2=calc_accuracy((">6",0),check_data)[1]
        
        if error_delete_1 < error_delete_2:
            tree_cut_new=("<=6", wrong_1)
            # use tree_cut_new to replace the original sub tree to finish pruning
            mytree_update_2=change_sub_tree(mytree_update,tree_cut,tree_cut_new)
            # if there is only one node of the newly-pruned tree, then stop pruning, else，use the newly pruned tree to replace the old tree
            if calc_leaf(mytree_update_2,0) > 2:
                mytree_update=mytree_update_2
            else:
                break
            accuracy_now=calc_accuracy(mytree_update,test_data_set)
            mytree_store=eval(str(mytree_update))
            accuracy_tree.append((accuracy_now[0],mytree_store))
            #print(calc_leaf(mytree_store,0))
            
        else:
            tree_cut_new=(">6",wrong_2)
            mytree_update_2=change_sub_tree(mytree_update,tree_cut,tree_cut_new)
            if calc_leaf(mytree_update_2,0) > 2:
                mytree_update=mytree_update_2
            else:
                break 
            accuracy_now=calc_accuracy(mytree_update,test_data_set)
            mytree_store=eval(str(mytree_update))
            accuracy_tree.append((accuracy_now[0],mytree_store))
            #print(calc_leaf(mytree_store,0))
    
    # use the pruned tree who has the largest accuracy to predict the second half of the train dataset as the final revised tree 
    for i in accuracy_tree:
        accuracy.append(i[0])
    accuracy.sort(reverse=True)
   
    for i in accuracy_tree:
        if i[0] == accuracy[0]:
            tree_best=i[1]
    return tree_best 

if __name__ == "__main__":
    whole_data = load_data(r"train.csv")
    data_set=whole_data[:700]
    Tree = grow_tree(data_set)
    test_data_set=whole_data[700:]
    final_test=load_data(r"test.csv")
    print("Without pruning, accuracy of the test dataset is:", calc_accuracy(Tree,final_test)[0])
    # use the [0:700] data of the train dataset to grow the original tree
    # use the [700:] data of the train dataset to calculate accuracy of each pruned tree
    revised_tree=CCP_prune(Tree,test_data_set,data_set)
    # finally, use the test dataset to predict and calculate prediction accuracy
    print("After pruning, accuracy of the test data set is:",calc_accuracy(revised_tree,final_test)[0])
    #print(calc_leaf(revised_tree,0))
    #print(revised_tree)