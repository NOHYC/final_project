# 모듈 호출
import numpy as np
import cv2 as cv
import os
import imutils
import argparse
import os.path as ops
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

prevTime = 0
weights_path= "./weight2/tusimple_lanenet.ckpt"
CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='The image path or the src image save dir')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')




def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

args = init_args()

# VideoCapture 객체 정의

cap = cv.VideoCapture(args.video) # high.mp4 test2.mp4

########### YOLO 관련 ############
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image 더 빠른 결과 320, 더 정확한 결과 608

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f : classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net1 = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# 출력 레이어 가져오기
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# bounding box 그리기
def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    if classes:
        assert(classId < len(classes))
        label = '%s' % (classes[classId])

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# NMS를 적용하여 낮은 confidence의 bounding box 제거
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        rec.append((left, top, width, height))


def color_extraction(img, num) :
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환

    lower_blue = (num-10, 100, 100) # hsv 이미지에서 바이너리 이미지로 생성, num = 90 -> green, num = 0 -> red
    upper_blue = (num+10, 255, 255)
    img_mask = cv.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색

    img_result = cv.bitwise_and(img, img, mask = img_mask) 

    return img_result

############################################################################


############################# 감지 영역 선택
isDragging = False
gx0, gy0, gw, gh = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)


def onMouse(event, x, y, flags, param):
    global isDragging, gx0, gy0, gimg,gw,gh
    if event == cv.EVENT_LBUTTONDOWN:
        isDragging = True
        gx0 = x
        gy0 = y
    elif event == cv.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = gimg.copy()
            cv.rectangle(img_draw, (gx0, gy0), (x, y), blue, 2)
            cv.imshow('test', img_draw)
    elif event == cv.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            gw = x - gx0
            gh = y - gy0
            if gw > 0 and gh > 0:
                img_draw = gimg.copy()
                cv.rectangle(img_draw, (gx0, gy0), (x, y), red, 2)
                #cv2.imshow('img', img_draw)
                roi = gimg[gy0:gy0+gh, gx0:gx0+gw]
                
                cv.imshow('cropped', roi)
                cv.moveWindow('cropped', 0, 0)
                #cv2.imwrite('./cropped.png', roi)
                
            else:
                cv.imshow('test', gimg)
                print('drag should start from left-top side')
capture = cv.VideoCapture(args.video)

gx0_list,gy0_list,gw_list,gh_list =[],[],[],[]
gimg = []
ret, frame = capture.read()
frame = cv.resize(frame,(512,256))
gimg = frame.copy()
capture.release()
cv.destroyAllWindows()

for num,vec in enumerate(['left','center','right']):
    place = int(512*num/3)
    gimg2 = cv.putText(gimg,vec, ( place +50 ,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2).copy()
    cv.imshow('test',gimg2)
    
    cv.setMouseCallback('test', onMouse)
    cv.waitKey()
    cv.destroyAllWindows()

    gx0_list.append(gx0)
    gy0_list.append(gy0)
    gw_list.append(gw)
    gh_list.append(gy0)
    print(gx0, gy0, gw, gh)
    

print(gx0_list,gy0_list,gw_list,gh_list)


############################################ 감지영역 끝

# 동영상
fourcc = cv.VideoWriter_fourcc(*'DIVX')
# 프레임 너비/높이, 초당 프레임 수 확인
width = cap.get(cv.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv.CAP_PROP_FPS) # 또는 cap.get(5)
#print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))
out1 = cv.VideoWriter('output_video.mp4', fourcc, fps, (512, 256))


input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
net = lanenet.LaneNet(phase='test', cfg=CFG)

binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet') ## value error

postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
sess_config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=sess_config)


with tf.variable_scope(name_or_scope='moving_avg'):
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.SOLVER.MOVING_AVE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

pre_mask_image = np.zeros((256,512,3))
pre_mask_image = np.asarray(pre_mask_image,np.uint8)
count =0
with sess.as_default():
    saver.restore(sess=sess, save_path=weights_path)
    

    while cap.isOpened(): # cap 정상동작 확인
        ret, frame = cap.read()
        fsize = frame.shape
        rec =[]
        # 프레임이 올바르게 읽히면 ret은 True
        if not ret:
            print(ret)
            print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
            break
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False) # yolo 부분 시작

        #for (x, y, w, h) in blob : # full-body만 검출 가능
        net1.setInput(blob)
        outs = net1.forward(getOutputsNames(net1))
        postprocess(frame, outs)
        fszie = frame.shape

        frame = cv.resize(frame, (1280, 512) ,interpolation = cv.INTER_LINEAR)

        image_vis = frame.copy()
        image = cv.resize(frame, (512,256), interpolation=cv.INTER_LINEAR)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image / 127.5 - 1.0

        binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]} )
        
        postprocess_result = postprocessor.postprocess(
            min_area_threshold = 150,
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis)
 
        mask_image = postprocess_result['mask_image']
        frame = cv.resize(frame,(512,256),interpolation = cv.INTER_LINEAR)
        
        try:    
            mask_image = cv.resize(mask_image, (fsize[1],fsize[0]), interpolation=cv.INTER_LINEAR)
            for num in rec:
                cv.rectangle(mask_image,(num[0], num[1]), (num[0] + num[2],num[1]  + num[3]), (0, 255, 0), 3)
            mask_image = cv.resize(mask_image , (512,256))
            
            lane_square =[]
            for i in range(3):
                lane_square.append(mask_image[gy0_list[i]:gy0_list[i]+gh_list[i], gx0_list[i]:gx0_list[i]+gw_list[i],:])
            lane_threshold = 240
            tt=cv.addWeighted(mask_image[:, :, (2, 1, 0)],0.3,frame,0.7,0)
        
            
            if np.max(lane_square[1][:,:,1])> lane_threshold:
                cv.putText(tt,'INFRONT_CAR', (100,250), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

            if (np.max(lane_square[0][:,:,0])> lane_threshold) and (np.max(lane_square[2][:,:,0])> lane_threshold) != 1 :
                count +=1
                if count ==5:
                    cv.putText(tt,'CAUTION', (250,250), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
                    count = 0
            else:
                count = 0
            if (np.max(lane_square[0][:,:,0])< lane_threshold)and(np.max(lane_square[2][:,:,0])> lane_threshold) == True:
                cv.putText(tt,'RIGHTcontrol', (250,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

            if (np.max(lane_square[0][:,:,0])> lane_threshold)and(np.max(lane_square[2][:,:,0])< lane_threshold) == True:
                cv.putText(tt,'LEFTcontrol', (250,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

            cv.imshow('Otter', tt)

        except Exception as e:
            cv.imshow('Otter', frame)   
        
        out1.write(tt)
        pre_mask_image = mask_image
        if cv.waitKey(1) == ord('q'):
            break
sess.close()
# 작업 완료 후 해제
cap.release()
out1.release()

cv.destroyAllWindows()