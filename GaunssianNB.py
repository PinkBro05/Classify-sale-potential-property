import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_visualisation import plot_data, qq_plot_data

from sklearn.linear_model import LogisticRegression

def main():
    # Load and preprocess data
    data = pd.read_csv('Classification_sell_potential/Reformatted.csv')
    new_data = data

    # Data Preprocessing
    le = LabelEncoder()
    new_data['label'] = le.fit_transform(new_data['Status'])
    new_data['RE Agency'] = le.fit_transform(new_data['RE Agency'])
    new_data['PropType'] = le.fit_transform(new_data['PropType'])
    new_data['Property Age'] = 2024 - new_data['Built Year']  # Assuming data is from 2024

    # Feature and target 
    features = ['PropType', 'Price', 'CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Median Price', 'Median Rental', 'PropertyCount'] # Remove 'RE Agency'
    x = new_data[features]
    y = new_data['label']

    # Data Visualization
    # plot_data(new_data)
    qq_plot_data(new_data)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    }

    # Initialize the model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Set up Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5)

    # Fit the Grid Search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Evaluate the tuned model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Sample Output
    print("Sample Output")
    print(data.iloc[151])

    predicted = best_model.predict(X_train[151].reshape(1, -1))
    print(f"Predicted: {predicted}")

if __name__ == '__main__':
    main()