from fairness.approximatemultiwasserstein import BaseHelper, WassersteinNoBin, MultiWasserStein
import numpy as np
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(123)

size = 10000

age = np.random.randint(18, 66, size)

gender = np.random.choice(['m', 'w', 'nb'], size)

nb_child = np.random.choice([0, 1, 2], size)

# Générer les salaires en fonction du genre en suivant les lois gamma spécifiées
salaries = []
for g in gender:
    if g == 'm':
        salary = np.random.gamma(20, 0.5)
    elif g == 'w':
        salary = np.random.gamma(2, 2)
    else:
        salary = np.random.gamma(0.5, 6)
    salaries.append(salary)

for i, n in enumerate(nb_child):
    if n == 0:
        salaries[i] += np.random.binomial(6, 0.75)
    elif g == 1:
        salaries[i] += np.random.binomial(5, 0.66)
    else:
        salaries[i] += np.random.binomial(1, 0.2)


# Créer le DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Nb_child': nb_child,
    'Income': salaries
})

x_ssa = df[['Gender', 'Nb_child']].to_numpy()
y_ssa = df['Income'].to_numpy()
x_ssa_calib, x_ssa_test, y_calib, y_test = train_test_split(
    x_ssa, y_ssa, test_size=0.3)

wst = MultiWasserStein()
wst.fit(y_calib, x_ssa_calib)
y_fair = wst.transform(y_test, x_ssa_test, epsilon=[0.1, 0.5])


def test_check_shape():
    x0 = np.zeros(67000)
    y0 = np.zeros(70000)
    with pytest.raises(ValueError):
        BaseHelper._check_shape(y0, x0)

    y1 = np.full(67000, 'a', dtype='str')
    with pytest.raises(ValueError):
        BaseHelper._check_shape(y1, x0)

    y2 = [0]*67000
    with pytest.raises(ValueError):
        BaseHelper._check_shape(y2, x0)

    x1 = [0]*70000
    with pytest.raises(ValueError):
        BaseHelper._check_shape(y0, x1)


def test_check_epsilon():
    epsilon = 8
    with pytest.raises(ValueError):
        BaseHelper._check_epsilon(epsilon)


def test_check_epsilon_size():
    wst0 = MultiWasserStein()
    wst0.fit(y_calib, x_ssa_calib)
    with pytest.raises(ValueError):
        wst0.transform(y_test, x_ssa_test, epsilon=[0.1, 0.2, 0.8])


def test_check_mod():
    x_ssa_calib0 = x_ssa_calib[np.isin(x_ssa_calib, ['m', 'nb'])]
    x_ssa_test0 = x_ssa_test[np.isin(x_ssa_test, ['m', 'nb', 'w'])]
    sens_val_calib = list(set(x_ssa_calib0))
    sens_val_test = list(set(x_ssa_test0))
    with pytest.raises(ValueError):
        BaseHelper._check_mod(sens_val_calib, sens_val_test)

    x_ssa_calib1 = x_ssa_calib[np.isin(x_ssa_calib, ['m', 'w'])]
    x_ssa_test1 = x_ssa_test[np.isin(x_ssa_test, ['nb', 'w'])]
    sens_val_calib1 = list(set(x_ssa_calib1))
    sens_val_test1 = list(set(x_ssa_test1))
    with pytest.raises(ValueError):
        BaseHelper._check_mod(sens_val_calib1, sens_val_test1)
