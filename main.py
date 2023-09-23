from sklearn.decomposition import PCA

from DatasetGenerator import DatasetGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# generate data if it does not exist
data_generator = DatasetGenerator()
data_generator()


# get train and test sets
def get_train_and_test_data():
    with open('./dataset.csv') as f:
        X = []
        y = []
        lines = f.readlines()
        for i in range(1, len(lines)):
            split_data = lines[i].split(',')
            X.append(tuple(int(i) for i in split_data[:-1]))
            y.append(int(split_data[-1].replace('\n', '')))

        return train_test_split(X, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = get_train_and_test_data()

# linear regression

reg = LinearRegression()
reg.fit(X_train, y_train)

prediction_difference = (reg.predict(X_test) - y_test)
mean_squared_error = sum(prediction_difference ** 2) / len(prediction_difference)
print(f'Mean squared error: {mean_squared_error:.2f}')

# pca
reg_pca = LinearRegression()

scaler = StandardScaler()
pca = PCA(n_components=1)
pca.fit(X_train)
X_pca = pca.fit_transform(X_train)
pca.fit_transform(X_train)
reg_pca.fit(X_pca, y_train)
reg_pca.predict(pca.fit_transform(X_test))

prediction_difference = (reg_pca.predict(pca.fit_transform(X_test)) - y_test)
mean_squared_error = sum(prediction_difference ** 2) / len(prediction_difference)
print(f'Mean squared error for PCA: {mean_squared_error:.2f}')
