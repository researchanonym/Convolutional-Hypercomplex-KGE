class Data:
    def __init__(self, data_dir=None, reverse=True, tail_pred_constraint=False):
        """
        ****** reverse=True
        Double the size of datasets by including reciprocal/inverse relations.
        We refer Canonical Tensor Decomposition for Knowledge Base Completion for details.

        ****** tail_pred_constraint=True
        Do not include reciprocal relations into testing. Consequently, MRR is computed by only tail entity rankings.
        """
        self.info = {'dataset': data_dir,
                     'dataset_augmentation': reverse,
                     'tail_pred_constraint': tail_pred_constraint}

        self.train_data = self.load_data(data_dir, data_type="train", add_reciprical=reverse)
        self.valid_data = self.load_data(data_dir, data_type="valid", add_reciprical=reverse)
        if tail_pred_constraint:
            self.test_data = self.load_data(data_dir, data_type="test", add_reciprical=False)
        else:
            self.test_data = self.load_data(data_dir, data_type="test", add_reciprical=reverse)

        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                                                 if i not in self.train_relations] + [i for i in self.test_relations \
                                                                                      if i not in self.train_relations]

    @staticmethod
    def load_data(data_dir, data_type, add_reciprical=True):
        try:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split() for i in data]
                if add_reciprical:
                    data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        except FileNotFoundError:
            raise FileNotFoundError(f'Please be sure that file located in {data_dir}')
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
