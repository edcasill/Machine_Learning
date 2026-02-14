import pandas as pd
import jax
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
    
    def extract_data(self,):
        """
        Extract and separate the data from the dataset

        :param self: Description
        """
        query = """
        SELECT e.emp_no,
               e.first_name,
               e.last_name,
               e.gender,
               e.hire_date,
               s.salary,
               s.to_date AS end_data,
               t.title
        FROM employees e
        JOIN salaries s ON e.emp_no = s.emp_no
        JOIN titles t ON e.emp_no = t.emp_no
        WHERE s.to_date = '9999-01-01' 
        AND t.to_date = '9999-01-01'
        """

        self.employee_ds = pd.read_sql(query, self.conn)
        print(self.employee_ds.head())

        return self



def main():
    database = ['barry', 'C@put_Dr@con1s!', 'localhost', 'employees']
    model = linear_model(database)

    model.extract_data()
    # model.separate_type_data()
    # model.augmented_X()
    # model.calculate_beta()
    # predict = model.linear_regression()
    # print('_'*60)
    # print(f'Prediction for dataset is: {predict}')


if __name__ == "__main__":
    main()