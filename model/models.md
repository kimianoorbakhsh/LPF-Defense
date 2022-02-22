# Models
## Data
The ModelNet40 data we used in our paper is uploaded in folder `Data/`. Any data used for training or testing should be in the same folder.
## Pointnet and Pointnet++
To train the Pointnet model, use the following code:
```python
python train_classification.py --model pointnet/pointnet2_cls_ssg --num_category 40 --log_dir {MODEL SAVE NAME} --main_path {PATH TO THIS DIRECTORY} --train_data_path {PATH TO TRAINING DATA} --train_label_path {PATH TO TRAINING LABELS} --test_data_path {PATH TO TESTING DATA} --test_label_path {PATH TO TESTING LABELS}
```
Please note that the training and testing data should be in .npy format.

The trained model will be save in a folder named `log/classification/{MODEL SAVE NAME}/`.

To test the Pointnet model, use the following code:
```python
python test_classification.py --model pointnet/pointnet2_cls_ssg --num_category 40 --log_dir {MODEL SAVE NAME} --main_path {PATH TO THIS DIRECTORY} --test_data_path {PATH TO TESTING DATA} --test_label_path {PATH TO TESTING LABELS}
```
# DGCNN
To train the DGCNN model, modify the code in `dgcnn.py` to use the DGCNN model. Modify the path of these variables: `train_data`, `train_labels`, `test_data`, `test_labels`.

To teest the DGCNN model, uncomment the code in `dgcnn.py` and modify the path of these variables: `test_data`, `test_labels`.