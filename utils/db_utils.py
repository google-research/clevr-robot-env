import lmdb
import pickle
import os
from torch.utils.data import Dataset

class LLMDataset(Dataset):

    def __init__(self, raw_path, data_dict, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_processed.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None

        if not (os.path.exists(self.processed_path)):
            self._process(data_dict)

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self, data_dict):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )

        with db.begin(write=True) as txn:
            for key, value in data_dict.items():
                serialized_key = pickle.dumps(key)
                serialized_value = pickle.dumps(value)
                txn.put(serialized_key, serialized_value)

        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        if self.transform is not None:
            data = self.transform(data)
        return data
