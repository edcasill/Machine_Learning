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
    
    def debug_titles(self):
        """Imprime todos los títulos únicos en la BD"""
        query = "SELECT DISTINCT title FROM titles WHERE to_date = '9999-01-01';"
        titles_df = pd.read_sql(query, self.conn)
        print("\n--- Títulos Disponibles (Copia uno de estos) ---")
        # Imprimimos entre comillas para ver si hay espacios extra
        for t in titles_df['title']:
            print(f"'{t}'")
        print("------------------------------------------------")

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
        current_date = pd.Timestamp.now()
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
    
    def calculate_beta(self):
        """
        Calculates the beta weigths

        :param self: Description
        """
        XtX = self.X_aug.T @ self.X_aug  # @ is the dot product, and is equal to use jnp.dot()
        Xty = self.X_aug.T @ self.y
        # linalg is linear algebra. It uses decomposition to solve the equation instead to use the direct inv
        self.beta = jnp.linalg.solve(XtX, Xty)
        # print(self.beta)

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
    hire = '1990-06-15'
    gender = 'M'
    title = 'Senior Engineer'
    # title = 'Staff'
    # title = 'Engineer'

    model = linear_model(database)
    #model.debug_titles()
    model.extract_data(title)
    model.separate_type_data()
    model.augmented_X()
    model.calculate_beta()
    predict = model.linear_regression(hire, gender)
    print('_'*60)
    print(f'Prediction for dataset is: {predict} as {title}')


if __name__ == "__main__":
    main()