# RangeLoss-Caffe
My implement of &lt;Range Loss for Deep Face Recognition with Long-tail>

# Note

* This layer has not been tested yet, there maybe exist some errors.
* After I train the network successfully, I will public the training files.

If you want to train with range loss, please add:

```c
message RangeLossParameter {
  optional int32 choose_k = 1 [default = 2];
  optional float inter_weight = 2 [default = 10e-4];
  optional float intra_weight = 3 [default = 10e-5];
  optional float margin = 4 [default = 2e4];
}
```
to the src/caffe/proto/caffe.proto

and:
```c
  optional RangeLossParameter range_loss_param = 155;
```
to the end of "message LayerParameter" 

The UniformLayer is used to get the uniform training data from filelist

please add:
```c
optional uint32 uniform_num = 14 [default = 1];
```
to the end of "message ImageDataParameter"

# Note

The implementation is a little different from the original formula in the paper,
I will check them again, welcome to dissucess.
