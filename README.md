# 这是一个基于luna16的含有语义分割和三维卷积判断肿瘤良性恶性的项目
./src/dataImprovment.py 是对三维肿瘤进行数据增强，将良性肿瘤输出到./data/benign_cube/，将恶性肿瘤输出到./data/tumor_cube/.  cube为[64,64,64]
运行方法(shell$) ： python3 ./src/dataImprovment.py 

./src/generateMask.py是对二维ct影像的切片进行数据增强，将mask输出到./data/mask/,将对应的图片输出到./data/candidate_image/. img为[512,512]
运行方法(shell$) : python3 ./src/generateMask.py

./src/main.py是对良性恶性肿瘤的模型resnet进行训练
运行方法(shell$) : python3 ./src/main.py 或 ./run.sh

./src/unet_train.py是对二维语义分割模型unet进行训练
运行方法(shell$) : python3 ./src/unet_train.py

目前状态 ： 肿瘤检测准确率98.5%,unet语义分割loss为0.002


![image](https://github.com/user-attachments/assets/1c2e7303-4105-4bf8-ae55-c12f6c6f43d2)
