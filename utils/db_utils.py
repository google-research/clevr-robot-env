import lmdb
import pickle
import os
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

def create_db(raw_path, data_dict, force_rewrite = False):
    raw_path = raw_path.rstrip('/')
    db_path = os.path.join(os.path.dirname(raw_path), os.path.basename(raw_path) + '_processed.lmdb')
    
    if (os.path.exists(db_path)) and (not force_rewrite):
        raise FileExistsError("The DB path specified already exists. You have specified initialization data for it. Proceeding would risk re-writing the DB existing there. If you do not want that DB, please delete it yourself first. Aborting now.")
    
    Path(os.path.dirname(raw_path)).mkdir(parents=True, exist_ok=True)

    db = lmdb.open(
            db_path,
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
    
def save_images(raw_path, data_dict):
    rgbs = []
    task_name = os.path.basename(raw_path)
    
    for idx in range(len(data_dict)): 
        rgbs.append(data_dict[idx]['image'])
    
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        
    for idx, rgb in enumerate(rgbs):
        image_path = os.path.join(raw_path, f'{task_name}_scene_{idx}.png')
        im = Image.fromarray(rgb)
        im.save(image_path)
        data_dict[idx]['image_path'] = f'{raw_path}/{task_name}_scene_{idx}.png'


class LLMDataset(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        # self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_processed.lmdb')
        self.transform = transform
        self.db = None
        self.keys = None

        # if db path does not exist, raise an error
        if not (os.path.exists(self.processed_path)):
            raise FileNotFoundError("DB path specified does not exist.")
            
    def _connect_db(self):
        # Establish read-only database connection
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
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
