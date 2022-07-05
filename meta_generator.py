import os.path as osp
import os
import pickle as pkl


class MetaGenerator():
    def __init__(self, pkl_path, meta_path='./list', name='train'):
        self.pkl = pkl_path
        self.name = name
        with open(self.pkl, 'rb') as cache_file:
            self.data_infos = pkl.load(cache_file)
        self.meta_path = meta_path
        os.makedirs(meta_path, exist_ok=True)
        self.write_meta_file(self.data_infos)
        

    def write_meta_file(self, data_infos):
        if 'mask_path' in self.data_infos[0].keys():
            with open(osp.join(self.meta_path, f'{self.name}_gt.txt'), 'w') as f:
                for data_info in data_infos:
                    path_str = data_info['img_path']
                    mask_str = data_info['mask_path']
                    exist_str = ' '.join([str(x) for x in list(data_info['lane_exist'])])
                    all_str = ' '.join([path_str, mask_str, exist_str])
                    f.write(all_str + '\n')
        with open(osp.join(self.meta_path, f'{self.name}.txt'), 'w') as f:
            for data_info in data_infos:
                path_str = data_info['img_path']
                f.write(path_str + '\n')

if __name__ == '__main__':
    mg = MetaGenerator('/home/sstl/LaneDetection/Data/MockLane/test.pkl', name='test')


