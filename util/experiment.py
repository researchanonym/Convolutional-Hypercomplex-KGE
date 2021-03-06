import json
from util.helper_funcs import *
from util.helper_classes import HeadAndRelationBatchLoader
from models.quat_models import *
from models.octonian_models import *
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Fixing the random seeds.
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


class Experiment:
    """
    Experiment class for training and evaluation
    """

    def __init__(self, *, dataset, model, parameters, ith_logger, store_emb_dataframe=False):

        self.dataset = dataset
        self.model = model
        self.store_emb_dataframe = store_emb_dataframe

        self.embedding_dim = parameters['embedding_dim']
        self.num_of_epochs = parameters['num_of_epochs']
        self.learning_rate = parameters['learning_rate']
        self.batch_size = parameters['batch_size']
        self.label_smoothing = parameters['label_smoothing']
        self.num_of_workers = parameters['num_workers']
        self.optimizer = None
        self.entity_idxs, self.relation_idxs, self.scheduler = None, None, None

        self.negative_label = 0.0
        self.positive_label = 1.0

        # Algorithm dependent hyper-parameters
        self.kwargs = parameters
        self.kwargs['model'] = self.model

        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name=self.model + ith_logger, p=self.storage_path)
        self.cuda = torch.cuda.is_available()
        if 'norm_flag' not in self.kwargs:
            self.kwargs['norm_flag'] = False

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    @staticmethod
    def get_re_vocab(data):
        # relation and tail entity
        re_vocab = defaultdict(list)
        for triple in data:
            re_vocab[(triple[1], triple[2])].append(triple[0])
        return re_vocab

    def get_batch_1_to_N(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.ones((len(batch), len(self.dataset.entities))) * self.negative_label
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = self.positive_label
        return np.array(batch), torch.FloatTensor(targets)

    def evaluate_one_to_n(self, model, data, log_info='Evaluate one to N.'):
        """
         Evaluate model
        """
        self.logger.info(log_info)
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch_1_to_N(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward_head_batch(e1_idx=e1_idx, rel_idx=r_idx)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)

        hit_1 = sum(hits[0]) / (float(len(data)))
        hit_3 = sum(hits[2]) / (float(len(data)))
        hit_10 = sum(hits[9]) / (float(len(data)))
        mean_rank = np.mean(ranks)
        mean_reciprocal_rank = np.mean(1. / np.array(ranks))

        self.logger.info(f'Hits @10: {hit_10}')
        self.logger.info(f'Hits @3: {hit_3}')
        self.logger.info(f'Hits @1: {hit_1}')
        self.logger.info(f'Mean rank: {mean_rank}')
        self.logger.info(f'Mean reciprocal rank: {mean_reciprocal_rank}')

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MR': mean_rank, 'MRR': mean_reciprocal_rank}

        return results

    def eval(self, model):
        """
        trained model
        """
        if self.dataset.test_data:
            results = self.evaluate_one_to_n(model, self.dataset.test_data,
                                             'Standard Link Prediction evaluation on Testing Data')
            with open(self.storage_path + '/results.json', 'w') as file_descriptor:
                num_param = sum([p.numel() for p in model.parameters()])
                results['Number_param'] = num_param
                results.update(self.kwargs)
                json.dump(results, file_descriptor)

    def train(self, model):
        """ Training."""
        if self.cuda:
            model.cuda()
        model.init()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.logger.info("{0} starts training".format(model.name))
        num_param = sum([p.numel() for p in model.parameters()])
        self.logger.info("'Number of free parameters: {0}".format(num_param))
        # Store the setting.
        with open(self.storage_path + '/settings.json', 'w') as file_descriptor:
            json.dump(self.kwargs, file_descriptor)

        model = self.k_vs_all_training_schema(model)

        # Save the trained model.
        torch.save(model.state_dict(), self.storage_path + '/model.pt')
        # Save embeddings of entities and relations in csv file.
        if self.store_emb_dataframe:
            entity_emb, emb_rel = model.get_embeddings()
            # pd.DataFrame(index=self.dataset.entities, data=entity_emb.numpy()).to_csv(TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            pd.DataFrame(index=self.dataset.entities, data=entity_emb.numpy()).to_csv(
                '{0}/{1}_entity_embeddings.csv'.format(self.storage_path, model.name))
            pd.DataFrame(index=self.dataset.relations, data=emb_rel.numpy()).to_csv(
                '{0}/{1}_relation_embeddings.csv'.format(self.storage_path, model.name))

    def __create_indexes(self):
        self.entity_idxs = {self.dataset.entities[i]: i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]: i for i in range(len(self.dataset.relations))}

        self.kwargs.update({'num_entities': len(self.entity_idxs),
                            'num_relations': len(self.relation_idxs)})
        self.kwargs.update(self.dataset.info)

    def train_and_eval(self):
        """
        Train and evaluate phases.
        """
        self.__create_indexes()
        model = None
        model = None
        if self.model == 'OMult':
            model = OMult(self.kwargs)
        elif self.model == 'ConvO':
            model = ConvO(self.kwargs)
        elif self.model == 'QMult':
            model = QMult(self.kwargs)
        elif self.model == 'ConvQ':
            model = ConvQ(self.kwargs)
        elif self.model == 'OMultBatch':
            model = OMultBatch(self.kwargs)
        elif self.model == 'ConvOBatch':
            model = ConvOBatch(self.kwargs)
        elif self.model == 'QMultBatch':
            model = QMultBatch(self.kwargs)
        elif self.model == 'ConvQBatch':
            model = ConvQBatch(self.kwargs)
        else:
            print(self.model, ' is not valid name')
            raise ValueError

        self.train(model)
        self.eval(model)

    def k_vs_all_training_schema(self, model):
        self.logger.info('k_vs_all_training_schema starts')

        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        losses = []

        head_to_relation_batch = DataLoader(
            HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(train_data_idxs), num_e=len(self.dataset.entities)),
            batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

        # To indicate that model is not trained if for if self.num_of_epochs=0
        loss_of_epoch, it = -1, -1

        for it in range(1, self.num_of_epochs + 1):
            loss_of_epoch = 0.0
            # given a triple (e_i,r_k,e_j), we generate two sets of corrupted triples
            # 1) (e_i,r_k,x) where x \in Entities AND (e_i,r_k,x) \not \in KG
            for head_batch in head_to_relation_batch:  # mini batches
                e1_idx, r_idx, targets = head_batch
                if self.cuda:
                    targets = targets.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                self.optimizer.zero_grad()
                loss = model.forward_head_and_loss(e1_idx, r_idx, targets)
                loss_of_epoch += loss.item()
                loss.backward()
                self.optimizer.step()
            losses.append(loss_of_epoch)
        self.logger.info('Loss at {0}.th epoch:{1}'.format(it, loss_of_epoch))
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        model.eval()
        return model