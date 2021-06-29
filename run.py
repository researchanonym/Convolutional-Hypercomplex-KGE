from util.experiment import Experiment
from util.data import Data
datasets = ['FB15k-237', 'YAGO3-10','WN18RR','FB15k', 'WN18', 'UMLS', 'KINSHIP']
models = ['QMultBatch', 'OMultBatch', 'ConvQBatch', 'ConvOBatch']

# Training script.
for kg_root in datasets:
    for model_name in models:
        data_dir = 'KGs/' + kg_root + '/'
        config = {
            'num_of_epochs': None,
            'batch_size': None,
            'learning_rate': None,
            'label_smoothing': None,
            'num_workers': None,
        }
        if model_name in ['ConvQBatch']:
            config.update({'embedding_dim': None,
                           'input_dropout': None,
                           'hidden_dropout': None,
                           'feature_map_dropout': None,
                           'num_of_output_channels': None,
                           'norm_flag': None, 'kernel_size': None})
        elif model_name in 'ConvOBatch':
            config.update({'embedding_dim': None,
                           'input_dropout': None,
                           'hidden_dropout': None,
                           'feature_map_dropout': None,
                           'num_of_output_channels': None,
                           'norm_flag': None, 'kernel_size': None})
        elif model_name in ['QMultBatch']:
            config.update({'embedding_dim': None,
                           'input_dropout': None,
                           'hidden_dropout': None,
                           'norm_flag': None})
        elif model_name in ['OMultBatch']:
            config.update({'embedding_dim': None,
                           'input_dropout': None,
                           'hidden_dropout': None,
                           'norm_flag': None})
        else:
            print(model_name)
            raise ValueError

        dataset = Data(data_dir=data_dir)
        experiment = Experiment(dataset=dataset,
                                model=model_name,
                                parameters=config, ith_logger='_' + kg_root)

        experiment.train_and_eval()
