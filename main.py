from sklearn.datasets import make_classification
import pandas as pd
from genetic_algoirthm.GA import GenticAlgorithmFeatureSelection

#build sample dataset
x, y = make_classification(n_features=100, n_samples=1000)
columns = [f'f_{i}' for i in range(1, 101)]

features = pd.DataFrame(x, columns=columns)
target = pd.DataFrame(y, columns=['target'])

# run feature selection
GA = GenticAlgorithmFeatureSelection(features=features, target=target, population_size=100, elite_rate=0.5,
                                     fitness_alpha=0.55, tourn_size=25, no_generation=50)
GA.run()
