#  블랙 박스 영상에 Object detection과 Segmentation 적용 프로젝트

### 팀명 : Varchar(5)

### 팀원 : **김동현**, 노용철, **정성훈**, 홍세준



## 주제선정 및 배경

카메라로 들어온 영상 정보를 활용하여 Object Detection, segmentation으로 차량과 차선등의 주변 환경 detection을 한다.



# 프로젝트 임무 분담

1. object detection : 김동현
   1. YOLO v3 
   2. YOLO v3 tracking
2. lane detection : 노용철
   1. LANENET
3. reinforce learning : 정성훈
4. object detection , lane detection : 홍세준
5. tensorRT : 노용철


### 결과의 유용성

- 소형화된 LiDAR센서와 고화질의 카메라 모듈을 통해 프로토 타입의 자율주행차를 제작하여 현업에서의 활용 가능성을 기대해보고 이를 바탕으로 무인 배송차, 방범용 로봇 등 무인화 가능한 전반적인 산업에 응용할 수 있다. 또한 최근 이슈화 되고 있는 어린이 보호구역에서의 사고 예방 등 산업을 떠난 일상의 안전에도 기여할 것으로 예상된다.

### 프로젝트 결과

1. YOLO v3 

   1. 블랙박스 영상을 활용하여 차량 검출

      ![yolov3](README.assets/yolov3.gif)

      

   2. bounding box와 HSV 변환을 활용한 신호등 색깔 검출

      ![image-20201228173249710](README.assets/image-20201228173249710.png)

      

2. YOLO tracking

   1. bounding box 크기를 활용한 차량 속도 표시 ( 작아지면(멀어짐) 파란색, 커지면(가까워짐) 빨간색 ) 

      ![yolotrackinh](README.assets/yolotracking.gif)

3. lanenet

   1. 차선 검출 및 comment 적용

      ![lanenet](README.assets/lanenet.gif)

4. YOLO v3 + LANENET 

   1. 고속도로

      ![high](README.assets/lanenet_yolo2.gif)

   2. 도심

      ![city](README.assets/lanenet_yolo1.gif)

5. YOLO v4 + LANENET 

   1. 고속도로

      ![high](README.assets/yl1.gif)

   2. 도심

      ![city](README.assets/yl2.gif)

6. reinforce learning

   1. epicode 400

      ![reinforce](README.assets/reinforce1-1609145682544.gif)

   2. episode 600

      ![reinforce2](README.assets/reinforce2.gif)



블랙 박스 영상을 활용하여 YOLO v3와 LANENET을 적용







### **프로젝트 실행방법**

**필요한 것**

파이썬 3.6

tensorflow ==1.12 tensorflow-gpu == 1.12

(cuda 9.0 cuDNN 7.2)

numpy
tqdm
glog
easydict
tensorflow_gpu
matplotlib
opencv4
scikit_learn
loguru



1. lanenet weight 파일 압축풀기

   4\.Final_model/weight2/tusimple_lanenet.ckpt.zip

2. yolo weight 파일 분할 압축 풀기

   4\.Final_model/yoloweight

```
python3 final.py --video [영상 경로]
```

3. New YOLO v4 + LANENET

   RTX 2080 ti
  
   fps : 30
   
   using trt docker (docker pull nvcr.io/nvidia/tensorrt:20.06-py3)
   

   lanenet train link : https://github.com/NOHYC/lanenet_torch_onnx_trt
   
   yolo v4 train link : https://github.com/AlexeyAB/darknet

   convert code link : https://github.com/Tianxiaomo/pytorch-YOLOv4

   recommenad darknet->ONNX->TensorRT

   **inference**

   **python3 demo_trt.py <tensorRT_engine_file_yolo> <tensorRT_engine_file_lanenet> <input_image> <input_H> <input_W>**
   
   directory data [ tensorRT_engine_file_yolo, tensorRT_engine_file_lanenet, cfgFile, namesFile, videoFile ]
   
   example
   
   ```
   python3 demo_trt.py yolov4_1_3_512_512_static.trt /Lanenet.trt /video.mp4 512 512 /classes.names 3 /yolo_lane.mp4 
   ```

