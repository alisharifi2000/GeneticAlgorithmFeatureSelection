# GeneticAlgorithmFeatureSelection
Feature Selection with Genetic Algorithm published in [pypi](https://pypi.org/project/GeneticAlgorithmFeatureSelection/).

#### Installation
    pip install GeneticAlgorithmFeatureSelection
    
#### Example

The original example code can be found in [test.py](https://github.com/alisharifi2000/GeneticAlgorithmFeatureSelection/blob/main/tests/test.py).
    
    from sklearn.datasets import make_classification
    import pandas as pd
    from genetic_algoirthm.GA import GenticAlgorithmFeatureSelection
    
Define the sample classification dataset
    
    x, y = make_classification(n_features=100, n_samples=2500)

input data must be pandas dataframe. we split target and features.
    
    columns = [f'f_{i}' for i in range(1, 101)]
    features = pd.DataFrame(x, columns=columns)
    target = pd.DataFrame(y, columns=['target'])
    
run feature selection

     GA = GenticAlgorithmFeatureSelection(features=features, target=target, population_size=100, elite_rate=0.5,
                                          fitness_alpha=0.55, tourn_size=25, no_generation=50)
     
     GA.run()
     
see history 
     
    history = GA.history

    for generation, detail in history.items():
        print(f'Generation :{generation}')
        print(f'best score: {detail["best_score"]}')
        print(f'features: {detail["selected_features"]}')
        
find best score and features in last generation
    
    print(f'best score last generation :{GA.best_score}')
    print(f'feature selected in last generation: {GA.selected_features}')
