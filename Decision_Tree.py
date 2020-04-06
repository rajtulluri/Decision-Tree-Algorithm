"""Decision Tree Algorithm implemented in Python""""

df = pd.read_csv('golf.csv')
df_test = pd.read_csv('golf_test.csv')

class BTree_Node:
    """BTree Node class as Data Structure to store every Node of the Decision Tree and corresponding Data"""
    
    def __init__(self,data=None,parent=None):
        #Initializing all variables
        
        self.key = data #Dataset subset after split
        self.children = [] 
        self.parent = parent
        self.split_attr = '' #The attribute, split occurs at the current level
        self.split_criteria = '' #The criteria for the child on the next level (rule)
        self.target_distribution = {} #Distribution of dataset according to the target
        self.prediction = '' #predicted value by the node
    
    def add(self,node):
        #Adds child to the parent node
        
        self.children.append(node)

def target_entropy(data,target):
    #Calculating Entropy for the Target variable for Information Gain calculations
    
    catgs = data[target].unique() #Unique categories in the Target variable
    total = len(df)
    E=0 #Entropy
    
    for catg in catgs:
        
        count = np.count_nonzero((df[target] == catg))
        E += -(count/total)*(np.log2(count/total))
    
    return E

def infoGain_calc(col,target,data):
    #Calulating Information Gain for the submitted column against the target
    
    catgs = list(data[col].unique()) #Unique categories in column 
    target_catgs = data[target].unique() #Target categories
    E = 0 #Entropy of the column
    total = len(df)
    E_target = target_entropy(df,target) #Target Entropy
    
    for catg in catgs:
        
        total_catg = np.count_nonzero((data[col] == catg))
        E_split = 0 #Entropy for one category in the column
        
        for target_catg in target_catgs:
            
            count = np.count_nonzero((data[target] == target_catg) & (data[col] == catg))
            if count == total_catg or count == 0: #If the subset only contains one type of target category
                E_split += 0
            else:
                E_split += -(count/total_catg)*(np.log2(count/total_catg))
            
        E += (total_catg/total) * E_split
           
    return (abs(E - E_target),col)

def build_tree(node,columns,target,max_depth=0,level=0):
    #Building the Decision tree
    
    split_column = '' #Column on which split occurs
    
    if level == max_depth: 
        return
    
    if not node.children: #No children - Either root node or new child node
        
        data = node.key  #Subset of Dataset stored at that node
        info_gains = []
        
        for col in columns:
            #Calculating Information gain for all columns of the data
            
            info_gains.append(infoGain_calc(col,target,node.key))
            
        split_column = max(info_gains)[1] #Max Information Gain column is chosen
        unique_catgs = data[split_column].unique()
        
        for catg in unique_catgs:
            #For every category of the splitting column, a child is created
            
            new_node = BTree_Node(parent=node) 
            new_node.key = data[data[split_column]==catg]
            new_node.split_criteria = catg
            new_node.target_distribution = dict(new_node.key[target].value_counts())
            node.add(new_node)
    
    node.split_attr = split_column 
    columns.remove(split_column) #Removing column that has already been split on
    level += 1 #For every new level created
    
    for node_child in node.children:
        #For every child, we build the tree further
        
        if len(node_child.key[target].unique()) == 1:
            #If for the node child, the target column in subset data has one category only, then it is the leaf node            
            
            node_child.prediction = max(node_child.target_distribution)
            
        else:
            build_tree(node_child,columns,target,max_depth,level)

def predict(node,data):
    #Predict target by Decision tree given the data
    
    if node.children == []:
        #If the node has no children, then it is leaf node
        
        if node.prediction:
            #Leaf node having only one type of category of target in subset dataset
            
            return node.prediction
        else:
            #If leaf node has multiple categories of target, incase tree was stopped prematurely
            
            return max(node.target_distribution)
    
    for node_child in node.children:
        #Traverse down the tree based on the rules formed 
        
        if data[node.split_attr] == node_child.split_criteria:
            tmp = predict(node_child,data)
    
    return tmp

def test(root,data):
    #A wrapper function for the predict function, incase of multiple test points
    
    prediction = []
    
    for ind,row in data.iterrows():
        #Calling predict for every data point
        
        prediction.append(predict(root,row))
        
    return pd.Series(prediction)

def accuracy(actual,predicted):
    #Calculate the accurcy of the Decision Tree
    
    if len(actual) != len(predicted):
        print("Different lengths of actual and predicted")
    
    ctr = 0
    total = len(actual)
    
    for act,pred in zip(actual,predicted):
        
        if act == pred:
            ctr += 1
    
    return (ctr/total)*100

root = BTree_Node(data=df)
target = 'PLAY'
columns = ['Outlook','Temperature','Humidity','Windy']
build_tree(root,columns,target,3)
pred = test(root,df_test)
accuracy(df_test[target],pred)
