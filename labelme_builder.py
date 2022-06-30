from typing import Callable, List, Tuple, Union, Any
import cv2
from matplotlib.pyplot import draw
import numpy as np
from scipy import interpolate
import json
import os.path as osp


class LabelmeObj(object):
    def __init__(self, path:str, step:int=10) -> None:
        with open(path) as file:
            self.labelme = json.load(file)
        self.json_path = path
        self.step = step
        self.root = osp.dirname(path)
        self.jsonname = osp.basename(path).replace(".json", "")
    
    @property
    def shape(self):
        """
        cv2 type -> (y, x)
        ——————>x
        |
        V
        y
        """
        return self.labelme['imageHeight'], self.labelme['imageWidth']
    
    @property
    def path(self):
        """
        Default json and image in the same path.
        """
        return osp.join(self.root, self.labelme['imagePath'])
    
    @property
    def lanes(self):
        return [x['points'] for x in self.labelme['shapes']]
    
    @property
    def labels(self):
        return [x['label'] for x in self.labelme['shapes']]
    
    @property
    def labels_(self):
        return [int(x[-1]) for x in self.labels]

    @property
    def binary_label(self):
        return np.array([1 if f"lane{x+1}" else 0 for x in range(len(self.labels))])

    def _generate_linespace(self, start:Union[float, int], end:Union[float, int]) -> np.ndarray:
        x = np.arange(start, end, self.step)
        x = np.append(x, end)
        return x

    def _interpolate_linear(self, x:list, y:list) -> Callable:
        """Linear interpolate"""
        f_linear = interpolate.interp1d(x, y, kind='linear')
        return f_linear

    def _interpolate_b_spline(self, x:list, y:list, x_new:np.ndarray, der:int=0) -> np.ndarray:
        """B Spline interpolate"""
        tck = interpolate.splrep(x, y)
        y_bspline = interpolate.splev(x_new, tck, der=der)
        return y_bspline

    def _draw_curve(self, img:np.ndarray, pts:List[Tuple], color:tuple) -> np.ndarray:
        xs = pts[0]
        ys = pts[1]
        for i, (x,y) in enumerate(zip(xs,ys)):
            try:
                img = cv2.line(img, (int(x),int(y)),(int(xs[i+1]),int(ys[i+1])), color, thickness=10)
            except:
                pass
        return img
    
    def _show_img(self, img:np.ndarray) -> None:
        """Show image"""
        cv2.imshow("image", img)
        cv2.waitKey(0)

    def get_lane_interpolate(self, lane:list) -> Any:
        """Interpolate single lane"""
        pt_x = [x[0] for x in lane]
        pt_y = [x[1] for x in lane]
        interpolate_y = self._generate_linespace(pt_y[0], pt_y[-1])
        if len(pt_x) < 10:
            f_linear = self._interpolate_linear(pt_y, pt_x)
            return f_linear(interpolate_y).astype(int).astype(float), interpolate_y.astype(int).astype(float)
        else:
            x_bspline = self._interpolate_b_spline(pt_y, pt_x, interpolate_y, der=0)
            return x_bspline.astype(int).astype(int).astype(float), interpolate_y.astype(int).astype(float)
    
    def get_lanes_interpolate(self) -> List[np.ndarray]:
        """Interpolate all lanes"""
        new_lanes = []
        for lane in self.lanes:
            new_lanes.append(self.get_lane_interpolate(lane))
        return new_lanes
    
    def to_new_txt(self, suffix:str='.lines') -> None:
        """Write new txt label"""
        if osp.exists(osp.join(self.root, f"{self.jsonname}{suffix}.txt")):
            raise ValueError("You may delete original txt first.")
        else:
            lanes = self.get_lanes_interpolate()
            with open(osp.join(self.root, f"{self.jsonname}_{suffix}.txt"), 'w') as f:
                for lane in lanes:
                    for ind in range(len(lane[0])):
                        f.write(str(lane[0][ind]))
                        f.write(' ')
                        f.write(str(lane[1][ind]))
                        f.write(' ')
                    f.write('\n')
    
    def draw_seg_png(self, save:bool=False) -> None:
        """
        Save segmentation png, if not save, then show segmentation.
        """
        background = np.zeros((*self.shape, 3))
        lanes = self.get_lanes_interpolate()
        for ind, lane in enumerate(lanes):
            color = (self.labels_[ind], self.labels_[ind], self.labels_[ind])
            background = self._draw_curve(background, pts=lane, color=color)
        if save:
            cv2.imwrite(osp.join(self.root, f"{self.jsonname}_mask.png"), background)
        else:
            self._show_img(background)
    
    def draw_view_png(self) -> None:
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 125, 125)]
        background = cv2.imread(self.path)
        lanes = self.get_lanes_interpolate()
        for ind, lane in enumerate(lanes):
            for i in range(len(lane[0])):
                pt = (int(lane[0][i]), int(lane[1][i]))
                cv2.circle(background, pt, 1, color[ind], 4)
        cv2.imwrite(osp.join(self.root, f"{self.jsonname}_view.png"), background)
    
    def get_all_info(self) -> dict:
        """Get meta info like clrnet"""
        result = {}
        result['img_name'] = osp.basename(self.path)
        result['img_path'] = self.path
        result['mask_path'] = osp.join(self.root, f"{self.jsonname}_mask.png")
        result['lane_exist'] = self.binary_label
        result['lanes'] = [list(zip(*x)) for x in self.get_lanes_interpolate()]
        result['json_path'] = self.json_path
        return result
    
    def main(self) -> dict:
        self.draw_view_png()
        try:
            self.to_new_txt()
        except:
            pass
        if osp.exists(osp.join(self.root, f"{self.jsonname}_mask.png")):
            return self.get_all_info()
        else:
            self.draw_seg_png(save=True)
            return self.get_all_info()


if __name__ == '__main__':
    labelme = LabelmeObj("/home/sstl/LaneDetection/Data/MockLane/frames/2.json")
    res = labelme.main()
    print(res)

    




