import os.path as osp
import os
import pickle as pkl
from labelme_builder import LabelmeObj
from tqdm import tqdm

class CULaneTrainBuilder(object):
    def __init__(self, path:str, filetype:str='a', cacher:str='traincacher.pkl') -> None:
        assert filetype in ['a', 'w'], "a: append, w: write"
        self.data_infos = []
        self.cacher = cacher
        if filetype == 'a':
            if osp.exists(cacher):
                with open(cacher, 'rb') as cache_file:
                    self.data_infos = pkl.load(cache_file)
                    existed_json_files = [x['json_path'] for x in self.data_infos]
                    print(f"Detected {len(existed_json_files)} file in cacher.")
            else:
                existed_json_files = []
                print(f"There is no cacher file in {cacher}.")
        if osp.isdir(path):
            self.root = path
            self.json_files = [osp.join(self.root, x) for x in os.listdir(path) if x.endswith('json')]
            if filetype == 'a':
                self.json_files = filter(lambda x: x not in existed_json_files, self.json_files)
        else:
            self.root = osp.dirname(path)
            self.json_files = [path]
    
        self.labelmeobjs = [LabelmeObj(x) for x in self.json_files]
    
    def main(self) -> None:
        for obj in tqdm(self.labelmeobjs):
            self.data_infos.append(obj.main())
        with open(self.cacher, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)


class CULaneTestBuilder(object):
    def __init__(self, path:str, filetype:str='a', cacher:str='testcacher.pkl') -> None:
        assert filetype in ['a', 'w'], "a: append, w: write"
        self.data_infos = []
        self.cacher = cacher
        if filetype == 'a':
            if osp.exists(cacher):
                with open(cacher, 'rb') as cache_file:
                    self.data_infos = pkl.load(cache_file)
                    existed_img_path = [x['img_path'] for x in self.data_infos]
            else:
                print(f"There is no cacher file in {cacher}.")
                existed_img_path = []
        if osp.isdir(path):
            self.root = path
            self.img_files = [osp.join(self.root, x) for x in os.listdir(path) if x.endswith('jpg') or x.endswith('png') or x.endswith('jpeg')]
            if filetype == 'a':
                self.img_files = filter(lambda x: x not in existed_img_path, self.img_files)
        else:
            self.root = osp.dirname(path)
            self.img_files = [path]

    def main(self) -> None:
        for img in tqdm(self.img_files):
            result = {}
            result['img_name'] = osp.basename(img)
            result['img_path'] = img
            result['lanes'] = []
            self.data_infos.append(result)
        with open(self.cacher, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

if __name__ == '__main__':
    ctrb = CULaneTrainBuilder(path='/home/sstl/LaneDetection/Data/MockLane/frames_max/foggy_4.json')
    ctrb.main()
    # cteb = CULaneTestBuilder(path='/home/sstl/LaneDetection/Data/MockLane/frames_test')
    # cteb.main()