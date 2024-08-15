from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Eksik olan import eklendi

class MLModels:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(),
            'random_forest': RandomForestRegressor()
        }
    
    def fit(self, model_name):
        model = self.models[model_name]
        model.fit(self.X_train, self.y_train)
        return model
    
    def evaluate(self, model_name):
        model = self.fit(model_name)
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.plot_confusion_matrix(y_pred)
        
        return mse, r2
    
    def plot_confusion_matrix(self, y_pred):
        plt.figure(figsize=(10, 7))
        sns.heatmap(pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred}).round(2).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Confusion Matrix")
        plt.show()
    
    def linear_regression(self):
        return self.evaluate('linear_regression')
    
    def decision_tree(self):
        return self.evaluate('decision_tree')
    
    def random_forest(self):
        return self.evaluate('random_forest')
