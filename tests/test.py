from sklearn.datasets import make_classification
import pandas as pd
from src.genetic_algoirthm.GA import GenticAlgorithmFeatureSelection

#build sample dataset
x, y = make_classification(n_features=100, n_samples=2500)
columns = [f'f_{i}' for i in range(1, 101)]

# must be pandas dataframe
features = pd.DataFrame(x, columns=columns)
target = pd.DataFrame(y, columns=['target'])

# run feature selection
GA = GenticAlgorithmFeatureSelection(features=features, target=target, population_size=100, elite_rate=0.5,
                                     fitness_alpha=0.55, tourn_size=25, no_generation=50)
GA.run()

history = GA.history

for generation, detail in history.items():
    print(f'Generation :{generation}')
    print(f'best score: {detail["best_score"]}')
    print(f'features: {detail["selected_features"]}')