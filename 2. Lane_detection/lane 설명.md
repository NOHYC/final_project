먼저 다운받을 것들

파이썬 3.6

tensorflow ==1.12 --> 텐서 플로우에 맞는 tensorflow-gpu와 CUDA + cuDNN이 있음

numpy
tqdm
glog
easydict
tensorflow_gpu
matplotlib
opencv
scikit_learn
loguru



그래픽 카드가 있을 때!

따로받아야하는 것 

cuda 9.0

cuDNN 7.2

다운 받는 방법은 엔디비아에서 확인 (cuDNN 환경변수 추가)

GPU 2개 써야 제대로된 성능을 발휘한다.



GPU 1개 사용시 FPS 5정도 나옴



설명 파일 있는 위치에서 프롬프트 켜고 python lane_detection.py --video [비디오 이름] 

lanenet은 TUSIMPLE 데이터로 학습

