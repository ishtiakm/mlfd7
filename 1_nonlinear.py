import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def filter_relevant_data(X_train, y_train, X_test, y_test,digit):

    # Filter the training data for labels 1 and 5
    train_filter = np.isin(y_train, [digit])
    X_train_f = X_train[train_filter]
    y_train_f = y_train[train_filter]

    # Filter the test data for labels 1 and 5
    test_filter = np.isin(y_test, [digit])
    X_test_f = X_test[test_filter]
    y_test_f = y_test[test_filter]

    # Return the filtered data
    return X_train_f, y_train_f, X_test_f, y_test_f
# Function to calculate vertical symmetry
def vertical_symmetry(image):
    left_half = image[:, :8]  # Left half (first 8 columns)
    right_half = np.fliplr(image[:, 8:])  # Right half, flipped horizontally
    symmetry = np.mean(np.abs(left_half - right_half))  # Mean absolute difference
    return symmetry

# Function to calculate horizontal symmetry
def horizontal_symmetry(image):
    top_half = image[:8, :]  # Top half (first 8 rows)
    bottom_half = np.flipud(image[8:, :])  # Bottom half, flipped vertically
    symmetry = np.mean(np.abs(top_half - bottom_half))  # Mean absolute difference
    return symmetry

# Function to calculate the width (using a threshold to determine pixel activation)
def width(image, threshold=0.5):
    # Count non-zero pixels (greater than threshold) along the columns
    width = np.sum(np.max(image > threshold, axis=0))
    return width

def intensity(image):
    return np.mean(image)

def polynomial_transform_3rd_order(X):
    # 3rd order polynomial transformation function as defined previously
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_squared = np.square(x1)
    x2_squared = np.square(x2)
    x1_cubed = np.power(x1, 3)
    x2_cubed = np.power(x2, 3)
    x1_x2 = x1 * x2
    x1_squared_x2 = x1_squared * x2
    x1_x2_squared = x1 * x2_squared
    #x1_squared_x2_squared = x1_squared + x2_squared
    
    return np.column_stack((
        x1, x2, 
        x1_squared, x2_squared, 
        x1_cubed, x2_cubed, 
        x1_x2, 
        x1_squared_x2, x1_x2_squared
    ))

# Main function to apply feature transform
def feature_transform(X,func1,func2,order='linear'):

    # Initialize list to store the transformed features
    transformed_features = []

    # Loop over each image (flattened) in X
    for i in range(X.shape[0]):
        image = X[i].reshape(16, 16)  # Reshape the 256 feature vector to 16x16
        
        # Calculate the feature transform
        #vertical_symmetry = calculate_vertical_symmetry(image)
        #horizontal_symmetry = calculate_horizontal_symmetry(image)
        #width = calculate_width(image)
        
        # For simplicity, let's combine these features (you can choose how to combine them)
        # For this example, we'll just use the width as the feature, but you can customize this
        #feature_value = func(image)
        
        # Append the calculated feature to the list
        features=[func1(image),func2(image)]
        transformed_features.append(features)
    transformed_features=np.array(transformed_features)
    if order =='nonlinear':
            x1 = transformed_features[:, 0]#
            x2 = transformed_features[:, 1]#
            x1_squared = np.square(x1)#
            x2_squared = np.square(x2)#
            x1_cubed = np.power(x1, 3)#
            x2_cubed = np.power(x2, 3)#
            x1_x2 = x1 * x2#
            x1_squared_x2 = x1_squared * x2#
            x1_x2_squared = x1 * x2_squared#
            #x1_squared_x2_squared = x1_squared + x2_squared
            
            return np.column_stack((
                x1, x2, 
                x1_squared, x2_squared, 
                x1_cubed, x2_cubed, 
                x1_x2, 
                x1_squared_x2, x1_x2_squared, 
                
            ))
            
        

    # Convert the list to a numpy array and return
    return transformed_features

    

def scatter_plot(X_train1,X_train5):
    
    plt.plot(X_train1[:,0],X_train1[:,1],'bo')
    plt.plot(X_train5[:,0],X_train5[:,1],'rx')
    plt.xlabel("Horizontal Symmetry")
    plt.ylabel("Vertical Symmetry")
    plt.legend(['Digit 1','Digit 5'])
    plt.savefig("scatter.png")
    plt.show()

def plot_with_separation(X_train1,X_train5,best_weights,order='linear',alg='',dataset=''):
    plt.figure()
    caption='Decision boundary for '+alg+' on '+dataset+'data'
    if order=='nonlinear':
        
        # Plot original data points
        plt.plot(X_train1[:, 0], X_train1[:, 1], 'bo')
        plt.plot(X_train5[:, 0], X_train5[:, 1], 'rx')
        plt.xlabel("Horizontal Symmetry")
        plt.ylabel("Vertical Symmetry")
        
        # Determine plot range
        x_min, x_max = min(min(X_train1[:, 0]), min(X_train5[:, 0])), max(max(X_train1[:, 0]), max(X_train5[:, 0]))
        y_min, y_max = min(min(X_train1[:, 1]), min(X_train5[:, 1])), max(max(X_train1[:, 1]), max(X_train5[:, 1]))
        
        # Create a grid of points in the original feature space
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]  # Shape: (10000, 2)
        
        # Apply the 3rd-order polynomial transformation to the grid points
        grid_transformed = polynomial_transform_3rd_order(grid_points)
        
        # Add a column of ones to account for the bias term in `best_weights`
        grid_transformed = np.hstack((np.ones((grid_transformed.shape[0], 1)), grid_transformed))
        
        # Calculate the decision boundary (classification threshold at 0)
        Z = np.dot(grid_transformed, best_weights)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')
        
        # Add legend and display plot
        
        plt.title(caption)
        plt.xlabel("Horizontal Symmetry")
        plt.ylabel("Vertical Symmetry")
        plt.legend(['Digit 1', 'Digit 5', 'Decision Boundary'])
        img_name=alg+'_'+dataset+'_'+order+'.png'
        plt.savefig(img_name)
        plt.show() 
    else:
        plt.plot(X_train1[:,0],X_train1[:,1],'bo')
        plt.plot(X_train5[:,0],X_train5[:,1],'rx')
        plt.title(caption)
        plt.xlabel("Horizontal Symmetry")
        plt.ylabel("Vertical Symmetry")
        w=best_weights
        x_min=min(min(X_train1[:,0]),min(X_train5[:,0]))
        x_max=max(max(X_train1[:,0]),max(X_train5[:,0]))
        sep_x1=np.linspace(x_min,x_max,50)
        sep_x2 = - (w[1] / w[2]) * sep_x1 - (w[0] / w[2])
        plt.plot(sep_x1,sep_x2)
        plt.legend(['Digit 1','Digit 5','Separator'])
        img_name=alg+'_'+dataset+'_'+order+'.png'
        plt.savefig(img_name)
        plt.show()

   


def linear_regression(inputs, outcomes):
    augmented_inputs = np.hstack((np.ones((inputs.shape[0], 1)), inputs))
    target = outcomes
    augmented_inputs_T = np.transpose(augmented_inputs)
    product_matrix = np.dot(augmented_inputs_T, augmented_inputs)
    inverse_product = np.linalg.inv(product_matrix)
    pseudo_inverse = np.dot(inverse_product, augmented_inputs_T)
    coefficients = np.dot(pseudo_inverse, target)
    
    return coefficients

def single_pla_pass(points, labels, current_weights):
    updated_weights = np.array(current_weights).reshape(-1, 1)
    extended_points = np.hstack((np.ones((points.shape[0], 1)), points))
    projections = extended_points @ updated_weights
    misclassified_indices = np.where((projections.flatten() * labels) <= 0)[0]

    if misclassified_indices.size > 0:
        misclassified_idx = misclassified_indices[0]
        updated_weights += (labels[misclassified_idx] * extended_points[misclassified_idx]).reshape(-1, 1)

    return updated_weights.flatten()

def calc_acc(weights, dataset, return_count=False):
    correct_predictions = 0
    features, labels = dataset

    for idx, point in enumerate(features):
        if np.dot(weights, np.insert(point, 0, 1)) * labels[idx] > 0:
            correct_predictions += 1

    accuracy = float(correct_predictions) / len(features)
    return (accuracy, correct_predictions) if return_count else accuracy

def pocket(dataset, initial_weights, max_iteration=10000):
    highest_accuracy = 0
    optimal_weights = initial_weights
    features, labels = dataset
    count = 0

    while count < max_iteration:
        initial_weights = single_pla_pass(features, labels, initial_weights)
        current_accuracy = calc_acc(initial_weights, dataset)
        
        if current_accuracy > highest_accuracy:
            optimal_weights = initial_weights
            highest_accuracy = current_accuracy
        count += 1

    return optimal_weights

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, Y, learning_rate=0.01, iterations=1000):
    # Add bias term (column of ones) to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Shape: (1500, 3)
    #print(f"X has a shape of {X.shape}")
    
    # Initialize weights
    W = np.zeros(X.shape[1])  # Shape: (3,)
    #print(f"W has a shape of {W.shape}")
    Y = (Y + 1) / 2  # Convert -1, 1 labels to 0, 1

    
    # Gradient descent
    for _ in range(iterations):
        # Predicted probabilities using the sigmoid function
        predictions = sigmoid(np.dot(X, W))  # Shape: (1500,)
    #    print(f"Pred has a shape of {predictions.shape}")
        
        # Compute the gradient
        gradient = np.dot(X.T, (predictions - Y)) / Y.size  # Shape: (3,)
        
        # Update weights
        W -= learning_rate * gradient
    
    return W
def logistic_regression_SGD(X, Y, learning_rate=0.01, iterations=1000, batch_size=32):
    # Add bias term (column of ones) to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Shape: (n_samples, n_features + 1)
    
    # Initialize weights
    W = np.zeros(X.shape[1])  # Shape: (n_features + 1,)
    Y = (Y + 1) / 2  # Convert -1, 1 labels to 0, 1

    # Mini-batch gradient descent
    n_samples = X.shape[0]
    for _ in range(iterations):
        # Shuffle the data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        # Loop over mini-batches
        for i in range(0, n_samples, batch_size):
            # Get mini-batch samples
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Predicted probabilities using the sigmoid function
            predictions = sigmoid(np.dot(X_batch, W))  # Shape: (batch_size,)

            # Compute the gradient for the mini-batch
            gradient = np.dot(X_batch.T, (predictions - Y_batch)) / batch_size  # Shape: (n_features + 1,)

            # Update weights
            W -= learning_rate * gradient

    return W

def linear_prog(X, Y, learning_rate=0.01, iterations=1000, lambda_C=1.0):
    """
    Perform logistic regression with L2 regularization.

    Parameters:
    - X: Input features, shape (n_samples, n_features)
    - Y: Labels, shape (n_samples,), should be 0 or 1
    - learning_rate: The learning rate for gradient descent
    - iterations: Number of iterations for gradient descent
    - lambda_C: Regularization parameter for L2 regularization

    Returns:
    - W: Learned weights, including bias term
    """
    # Add bias term (column of ones) to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Initialize weights
    W = np.zeros(X.shape[1])
    
    # Convert labels from -1, 1 to 0, 1 if necessary
    Y = (Y + 1) / 2

    # Gradient descent
    for _ in range(iterations):
        # Predicted probabilities using the sigmoid function
        predictions = sigmoid(np.dot(X, W))
        
        # Compute the gradient with L2 regularization
        gradient = (np.dot(X.T, (predictions - Y)) / Y.size) + lambda_C * W
        
        # Update weights
        W -= learning_rate * gradient
    
    return W


# Load the data from the file
def data_preprocess(file_path):
    data = []
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.split()))  # Convert each line to a list of floats
            data.append(values)
    
    # Convert the list to a NumPy array for easier manipulation
    data = np.array(data)
    
    # Split the data into labels (first column) and features (remaining columns)
    labels = data[:, 0]      # Labels are the first column
    features = data[:, 1:]   # Features are the remaining columns
    return features,labels

# Path to the ZipDigits.train file
train_file_path = 'ZipDigits.train'
test_file_path = 'ZipDigits.test'

train_features,train_labels=data_preprocess(train_file_path)
test_features,test_labels=data_preprocess(test_file_path)

X_train_f1,y_train_f1,X_test_f1,y_test_f1=filter_relevant_data(train_features,train_labels,test_features,test_labels,1)
X_train_f5,y_train_f5,X_test_f5,y_test_f5=filter_relevant_data(train_features,train_labels,test_features,test_labels,5)
y_train_f5=np.where(y_train_f5 == 5, -1, y_train_f5)
y_test_f5=np.where(y_test_f5 == 5, -1, y_test_f5)

X_train_new_1=feature_transform(X_train_f1,horizontal_symmetry,vertical_symmetry,order='nonlinear')
X_train_new_5=feature_transform(X_train_f5,horizontal_symmetry,vertical_symmetry,order='nonlinear')
X_test_new_1=feature_transform(X_test_f1,horizontal_symmetry,vertical_symmetry,order='nonlinear')
X_test_new_5=feature_transform(X_test_f5,horizontal_symmetry,vertical_symmetry,order='nonlinear')

X_train_2=np.concatenate((X_train_new_1, X_train_new_5), axis=0)
X_test_2=np.concatenate((X_test_new_1, X_test_new_5), axis=0)
Y_train_2=np.concatenate((y_train_f1, y_train_f5))
Y_test_2=np.concatenate((y_test_f1, y_test_f5))

weights=linear_regression(X_train_2, Y_train_2)
w_pp=pocket([X_train_2,Y_train_2], weights, max_iteration = 10000)
w_log=logistic_regression(X_train_2,Y_train_2,learning_rate=0.002, iterations=500000)
w_sgd=logistic_regression_SGD(X_train_2,Y_train_2, learning_rate=0.001, iterations=10000,batch_size=1)
w_lp=linear_prog(X_train_2,Y_train_2, learning_rate=0.01, iterations=100000, lambda_C=0.001)
print("calculating Ein:")
print()
print(f"Ein for PLA with pocket is {(1-calc_acc(w_pp, [X_train_2,Y_train_2]))}")
print(f"Ein for logistic regression with GD is {(1-calc_acc(w_log, [X_train_2,Y_train_2]))}")
print(f"Ein for logistic regression with SGD is {(1-calc_acc(w_sgd, [X_train_2,Y_train_2]))}")
print(f"Ein for linear programming with pocket is {(1-calc_acc(w_pp, [X_train_2,Y_train_2]))}")
print()

print("calculating Etest:")
print()
print(f"Etest for PLA with pocket is {(1-calc_acc(w_pp, [X_test_2,Y_test_2]))}")
print(f"Etest for logistic regression with GD is {(1-calc_acc(w_log, [X_test_2,Y_test_2]))}")
print(f"Etest for logistic regression with SGD is {(1-calc_acc(w_sgd, [X_test_2,Y_test_2]))}")
print(f"Etest for linear programming with pocket is {(1-calc_acc(w_pp, [X_test_2,Y_test_2]))}")
print()

print("plotting decision boundary on training data")
plot_with_separation(X_train_new_1,X_train_new_5,w_pp,order='nonlinear',alg='PLA with pocket',dataset='train')
plot_with_separation(X_train_new_1,X_train_new_5,w_log,order='nonlinear',alg='Logistic regression with GD',dataset='train')
plot_with_separation(X_train_new_1,X_train_new_5,w_sgd,order='nonlinear',alg='Logistic regression with SGD',dataset='train')
plot_with_separation(X_train_new_1,X_train_new_5,w_lp,order='nonlinear',alg='Logistic regression with Linear Programming',dataset='train')

print("plotting decision boundary on test data")
plot_with_separation(X_test_new_1,X_test_new_5,w_pp,order='nonlinear',alg='PLA with pocket',dataset='test')
plot_with_separation(X_test_new_1,X_test_new_5,w_log,order='nonlinear',alg='Logistic regression with GD',dataset='test')
plot_with_separation(X_test_new_1,X_test_new_5,w_sgd,order='nonlinear',alg='Logistic regression with SGD',dataset='test')
plot_with_separation(X_test_new_1,X_test_new_5,w_lp,order='nonlinear',alg='Logistic regression with Linear Programming',dataset='test')
