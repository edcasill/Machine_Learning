import pandas as pd
import jax
import jax.numpy as jnp


class linear_model:
    def __init__(self, cvs):
        # read csv
        self.insurance_data = pd.read_csv(cvs)

    def extract_data(self, sb=False):
        """
        Extract and separate the data from the dataset

        :param self: Description
        :param sb: add a new characteristic to the model, add importance to bmi on smoker persons
        """
        # separate de classification data from the numerical
        self.insurance_data['sex'] = self.insurance_data['sex'].map({'female': 0, 'male': 1})
        self.insurance_data['smoker'] = self.insurance_data['smoker'].map({'no': 0, 'yes': 1})
        if sb:
            # by adding this column the estimation is better, going from 0.75 to 0.85
            self.insurance_data['smoker_bmi'] = self.insurance_data['smoker'] * self.insurance_data['bmi']
        self.insurance_data = pd.get_dummies(self.insurance_data, columns=['region'])
        print(self.insurance_data)

        return self

    def separate_type_data(self):
        """
        Separate the objective value (charges) from the input data

        :param self: Description
        """
        # get objective value (charges)
        self.X_wo_charges = self.insurance_data.drop('charges', axis=1).astype(float).values
        self.y_charges = self.insurance_data['charges'].astype(float).values

        return self

    def augmented_X(self):
        """
        Add beta_0 to the vector to add the property of no singular to the transpose matrix

        :param self: Description
        """
        mid_vector = jnp.ones((self.X_wo_charges.shape[0], 1))
        self.X_aug = jnp.concatenate([mid_vector, jnp.array(self.X_wo_charges)], axis=1)
        self.y = jnp.array(self.y_charges)
        print(self.X_aug.shape)

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
        print(self.beta)

        return self

    def linear_regression(self):
        """
        linear regression model

        :param self: Description
        :return: prediction for data, y_hat
        """
        # linear regression model
        self.y_hat = self.X_aug @ self.beta
        return self.y_hat

    def calculate_error(self):
        """
        Calcualates the error, MSE and RMSE to see how far are we from the optimal value

        :param self: Description
        """
        res = self.y - self.y_hat
        print(f'The average error is: {jnp.mean(jnp.abs(res))}')
        mse = jnp.mean(jnp.power(res, 2))
        rmse = jnp.sqrt(mse)
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')

        rss = jnp.sum(jnp.square(res))
        y_med = jnp.mean(self.y)
        ss_total = jnp.sum(jnp.square(self.y - y_med))
        r2 = 1 - (rss / ss_total)
        print(f'R² is : {r2}')

    def test_data(self, new_data, sb=False):
        """
        Test the imput data
        
        :param self: Description
        :param new_data: Description
        :param sb: Description
        """
        age = float(new_data[0])
        sex = 1 if new_data[1] == 'male' else 0
        bmi = float(new_data[2])
        children = float(new_data[3])
        smoker = 1 if new_data[4] == 'yes' else 0
        regions = ['northeast', 'northwest', 'southeast', 'southwest']
        region_encoded = [1.0 if new_data[5] == r else 0.0 for r in regions]

        # characteristic vector
        features = [1.0, age, float(sex), bmi, children, float(smoker)]

        if sb:
             smoker_bmi = float(smoker * bmi)
             features.append(smoker_bmi)
        features = features + region_encoded
        X_new_aug = jnp.array(features)
        prediction = jnp.dot(X_new_aug, self.beta)

        return prediction


def main():
    smoker_bmi = True
    model = linear_model("insurance.csv")

    model.extract_data(sb=smoker_bmi)
    model.separate_type_data()
    model.augmented_X()
    model.calculate_beta()
    predict = model.linear_regression()
    print('_'*60)
    print(f'Prediction for dataset is: {predict}')
    model.calculate_error()

    print('_'*60)
    test = [27, 'male', 25, 0, 'no', 'southwest', 20000]
    test_result = model.test_data(test, sb=smoker_bmi)
    print(f'New data {test}')
    print(f'Insurance is: {test_result}')


if __name__ == "__main__":
    main()
