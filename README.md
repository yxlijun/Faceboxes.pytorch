## FaceBoxes: A CPU Real-time Face Detector with High Accuracy ##
[A PyTorch Implementation of FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/pdf/1708.05234.pdf)


### Description
I train faceboxes with pytorch which approachs the [official code](https://github.com/sfzhang15/FaceBoxes),the final model can be downloaded in [faceboxes.pytorch](https://pan.baidu.com/s/1dsd9FY5JjO0hvx2zsSMStQ), the ap in AFW,PASCAL_FACE and FDDB as following:

| 	AFW     |   PASCAL	|   FDDB   |
| --------- |-----------| ---------|
|	98.32   |    96.35  |  95.2	   |

### Requirement
* pytorch 0.3 
* opencv 
* numpy 
* easydict

### Prepare data 
1. download WIDER face dataset
2. modify data/config.py 
3. ``` python prepare_wider_data.py```

### Train 
``` 
python train.py --lr 0.001
```

### Evalution
according to yourself dataset path,modify data/config.py,the evalution way is same with the [official code](https://github.com/sfzhang15/FaceBoxes)
1. Evaluate on AFW.
```
python tools/afw_test.py
```
2. Evaluate on FDDB 
```
python tools/fddb_test.py
```
3. Evaluate on PASCAL  face 
``` 
python tools/pascal_test.py
```
4. Test image
```
python demo.py
```

### Result
<div align="center">
<img src="https://github.com/yxlijun/faceboxes.pytorch/blob/master/tmp/test2.jpg" height="200px" alt="demo" >
<img src="https://github.com/yxlijun/faceboxes.pytorch/blob/master/tmp/test.jpg" height="200px" alt="demo" >
</div>


### References
* [FaceBoxes](https://github.com/sfzhang15/FaceBoxes)
* [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/pdf/1708.05234.pdf)

