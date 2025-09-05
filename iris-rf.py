# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns




# import dagshub
# dagshub.init(repo_owner='campusx-official', repo_name='mlflow-dagshub-demo', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/campusx-official/mlflow-dagshub-demo.mlflow")

# #-----------------------------------------------------------------------------------------------
# import dagshub
# dagshub.init(repo_owner='aamir490', repo_name='mlflow-dagshub-demo', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/aamir490/mlflow-dagshub-demo.mlflow")

# #-------------------------------------------------------------------------------------------------


# # Load the iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the parameters for the Random Forest model
# max_depth = 1
# n_estimators = 100

# # apply mlflow

# mlflow.set_experiment('iris-rf')

# with mlflow.start_run():

#     rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

#     rf.fit(X_train, y_train)

#     y_pred = rf.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)

#     mlflow.log_metric('accuracy', accuracy)

#     mlflow.log_param('max_depth', max_depth)
#     mlflow.log_param('n_estimators', n_estimators)

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

#     mlflow.sklearn.log_model(rf, "random forest")

#     mlflow.set_tag('author','rahul')
#     mlflow.set_tag('model','random forest')

#     print('accuracy', accuracy)

#--------------------------------------------------------------------------------------------------
# ------------------------
# 1. Import Libraries
# ------------------------
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 2. Initialize DagsHub & MLflow
# ------------------------
import dagshub
dagshub.init(repo_owner='aamir490', repo_name='mlflow-dagshub-demo', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aamir490/mlflow-dagshub-demo.mlflow")

# ------------------------
# 3. Load Dataset
# ------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 4. Set Model Parameters
# ------------------------
max_depth = 1
n_estimators = 100

# ------------------------
# 5. Start MLflow Experiment
# ------------------------
mlflow.set_experiment('iris-rf')

with mlflow.start_run():

    # ------------------------
    # 5a. Train Random Forest
    # ------------------------
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    # ------------------------
    # 5b. Predict & Evaluate
    # ------------------------
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', accuracy)

    # ------------------------
    # 5c. Confusion Matrix
    # ------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save & log plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ------------------------
    # 5d. Log Script
    # ------------------------
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        print("⚠️ Skipping __file__ logging (not running from a .py file)")

    # ------------------------
    # 5e. Log Model & Tags
    # ------------------------
    mlflow.sklearn.log_model(rf, "random_forest")
    mlflow.set_tag('author', 'rahul')
    mlflow.set_tag('model', 'random_forest')

    print('Accuracy:', accuracy)
    print(" ✅ Random Forest model logged in MLflow ")
# ----------------------------------------------------------------------