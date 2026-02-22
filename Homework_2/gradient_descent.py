import pandas as pd
import jax
import jax.numpy as jnp
from sqlalchemy import create_engine
from urllib.parse import quote_plus


class linear_model:
    def __init__(self, dataset):
        user = dataset[0]
        password = quote_plus(dataset[1])
        host = dataset[2]
        db = dataset[3]
        url = f"mysql+pymysql://{user}:{password}@{host}/{db}"
        self.conn = create_engine(url)
        self.beta = None
    
    def extract_data(self, user_title):
        """
        Extract and separate the data from the dataset

        :param self: Description
        """
        query = """
        SELECT e.emp_no,
               e.gender,
               e.hire_date,
               s.salary,
               s.to_date AS end_data,
               t.title
        FROM employees e
        JOIN salaries s ON e.emp_no = s.emp_no
        JOIN titles t ON e.emp_no = t.emp_no
        WHERE s.to_date = '9999-01-01'
        """
        all_data = pd.read_sql(query, self.conn)
        # self.employee_ds = pd.read_sql(query, self.conn)
        self.employee_ds = all_data[all_data['title'].str.strip() == user_title.strip()]
        print(self.employee_ds.head())
        return self

    def separate_type_data(self):
        """
        Separate the objective value from the input data, also prepares the input data for the model

        :param self: Description
        """
        data_frame = self.employee_ds.copy()

        self.y_ini = data_frame['salary'].values.reshape(-1, 1) # Objective value

        # data prepared for X
        data_frame['gender_enc'] = data_frame['gender'].map({'F': 0, 'M': 1})
        data_frame['hire_date'] = pd.to_datetime(data_frame['hire_date'])
        current_date = pd.Timestamp("25-12-31")
        data_frame['years_experience'] = (current_date - data_frame['hire_date']).dt.days / 365.25

        self.X_raw = data_frame[['years_experience', 'gender_enc']].values
        return self
    
    def augmented_X(self):
        """
        Add beta_0 to the vector to add the property of no singular to the transpose matrix

        :param self: Description
        """
        mid_vector = jnp.ones((self.X_raw.shape[0], 1))
        self.X_aug = jnp.concatenate([mid_vector, jnp.array(self.X_raw)], axis=1)
        self.y = jnp.array(self.y_ini)
        # print(self.X_aug.shape)

        return self
    
    def gradient_descent(self, grid):
        """
        Calculates the gradient descent ridge
        
        :param self: Descripción
        """
        seed = jax.random.PRNGKey(73)
        best_lambda = None
        best_mse = float('inf')

        index = jnp.arange(len(self.X_aug))
        index_shuffle = jax.random.permutation(seed, index)
        X_shuffle = self.X_aug[index_shuffle]
        y_shuffle = self.y[index_shuffle]
        split_index = int(len(X_shuffle) * 0.9)

        X_training = X_shuffle[:split_index]
        y_training = y_shuffle[:split_index]
        X_test = X_shuffle[split_index:]
        y_test = y_shuffle[split_index:]

        n_features = X_training.shape[1]
        t = 5000  # epochs on update function
        n_samples = len(y_training)

        for val in grid:
            beta_temp = jax.random.normal(seed, (n_features, 1)) * 0.01
            for i in range(t):
                y = X_training @ beta_temp
                error = y_training - y
                gradient_desc = (-2 / n_samples) * (X_training.T @ error)
                beta_no_bias = beta_temp.at[0].set(0.0)
                gradient_reg = 2 * val * beta_no_bias
                beta_temp = beta_temp - 1e-7 * (gradient_desc + gradient_reg)

            y_pred = X_test @ beta_temp
            mse = jnp.mean((y_test - y_pred) ** 2)
            if jnp.isnan(mse):
                print(f"Lambda {val}: MSE is NaN")
                continue
            print(f"Lambda: {val}, MSE: {mse:.2f}")
            if mse < best_mse:
                best_mse = mse
                best_lambda = val
                self.beta = beta_temp
        print(f'Best lambda is: {best_lambda}')
        print(f'Best beta is: {self.beta}')
        return self
    
    def linear_regression(self, hire_date, gender):
        """
        linear regression model applied on a salary for a person in 2025

        :param self: Description
        :return: prediction for data, y_hat
        """
        target_date_2025 = pd.Timestamp("2025-12-31")
        h_date = pd.Timestamp(hire_date)
        years_exp_2025 = (target_date_2025 - h_date).days / 365.25
        gen_val = 1 if gender == 'M' else 0
        x_input = jnp.array([[1.0, years_exp_2025, gen_val]])
        prediction = x_input @ self.beta

        return prediction[0][0].item()


def main():
    database = ['barry', 'C@put_Dr@con1s!', 'localhost', 'employees']
    hire = '2022-06-15'
    gender = 'M'
    title = 'Engineer'
    grid = [0.0, 0.1, 1.0, 10.0, 50.0, 100.0]

    model = linear_model(database)
    model.extract_data(title)
    model.separate_type_data()
    model.augmented_X()
    model.gradient_descent(grid)
    predict = model.linear_regression(hire, gender)
    print('_'*60)
    print(f'Prediction for dataset is: {predict} as {title}')


if __name__ == "__main__":
    main()