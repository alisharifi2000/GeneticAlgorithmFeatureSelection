# GeneticAlgorithmFeatureSelection
Feature Selection with Genetic Algorithm published in [pypi](https://pypi.org/project/GeneticAlgorithmFeatureSelection/).

#### Installation
    from GeneticAlgorithmFeaturesSelection.GA import FeatureSelection
    
#### Example

The original example code can be found in [test.py](https://github.com/alisharifi2000/GeneticAlgorithmFeatureSelection/blob/main/tests/test.py).
    
    from sklearn.datasets import make_classification
    import pandas as pd
    from genetic_algoirthm.GA import GenticAlgorithmFeatureSelection
    
Define the sample classification dataset
    
    x, y = make_classification(n_features=25, n_samples=1500, n_classes=4, n_clusters_per_class=3,
                               n_informative=4)

input data must be pandas dataframe. we split target and features.
    
    columns = [f'f_{i}' for i in range(1, 26)]
    features = pd.DataFrame(x, columns=columns)
    target = pd.DataFrame(y, columns=['target'])
    
run feature selection

     FS = FeatureSelection(features=features, target=target, population_size=1000, elite_rate=0.25,
                           mut_rate=0.2, k_folds=4, fitness_alpha=0.85, tourn_size=30,
                           no_generation=120, method='DesicionTree', scoring='accuracy')
     FS.run()
     
see history 
     
    history = FS.history

    for generation, detail in history.items():
        print(f'Generation :{generation}')
        print(f'best score: {detail["best_score"]}')
        print(f'features: {detail["selected_features"]}')
        
find best score and features in last generation
    
    print(f'best score last generation :{GA.best_score}')
    print(f'feature selected in last generation: {GA.selected_features}')
