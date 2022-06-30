# CULanelike Dataset Builder

Build dataset like culane.

## Quick Start

### Install Labelme

```shell
pip install labelme
labelme
```

![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2022-06-29-092100.png)

### Label lane with fixed rules

* lane1: adjacent left lane line
* lane2: own lane left lane line
* lane3: own lane right lane line 
* lane4: adjacent right lane line
* Do not label lane beyond obstacles.
* Use line or linestrip.
* Put json file and image file in the same path.

 

![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2022-06-29-093234.png)

![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2022-06-29-093308.png)

### Build train dataset

```shell
python culane_builder/main.py --path=<your image file path> --cache=<output path>
```




