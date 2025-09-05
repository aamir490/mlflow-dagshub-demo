# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# import dagshub
# dagshub.init(repo_owner='campusx-official', repo_name='mlflow-dagshub-demo', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/campusx-official/mlflow-dagshub-demo.mlflow")

# # Load the iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the parameters for the Random Forest model
# max_depth = 1

# # apply mlflow

# mlflow.set_experiment('iris-dt')

# with mlflow.start_run():

#     dt = DecisionTreeClassifier(max_depth=max_depth)

#     dt.fit(X_train, y_train)

#     y_pred = dt.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)

#     mlflow.log_metric('accuracy', accuracy)

#     mlflow.log_param('max_depth', max_depth)

#     # Create a confusion matrix plot
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.title('Confusion Matrix')
    
#     # Save the plot as an artifact
#     plt.savefig("confusion_matrix.png")

#     # mlflow code
#     mlflow.log_artifact("confusion_matrix.png")

#     mlflow.log_artifact(__file__)

#     mlflow.sklearn.log_model(dt, "decision tree")

#     mlflow.set_tag('author','nitish')
#     mlflow.set_tag('model','decision tree')

#     print('accuracy', accuracy)

#---------------------------------------------------------------------    


import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------------------------------

# ALSO ADD THIS :
import dagshub
dagshub.init(repo_owner='aamir490', repo_name='mlflow-dagshub-demo', mlflow=True)


# CHANGE HERE : - https://dagshub.com/aamir490/mlflow-dagshub-demo.mlflow
mlflow.set_tracking_uri("https://dagshub.com/aamir490/mlflow-dagshub-demo.mlflow")

#-------------------------------------------------------------------------------------------------

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Decision Tree model
max_depth = 1

mlflow.set_experiment('iris-dt')

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy', accuracy)

    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the current script (works only in .py files)
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        print("⚠️ Skipping __file__ logging (not running from a .py file)")

    # Log the model
    mlflow.sklearn.log_model(dt, "decision_tree")

    # Set tags
    mlflow.set_tag('author','nitishaa')
    mlflow.set_tag('model','decision_tree')

    print('Accuracy:', accuracy)



















