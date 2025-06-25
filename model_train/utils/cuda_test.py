import tensorflow as tf

# GPU 장치 목록 출력
print("Usable GPU:")
print(tf.config.list_physical_devices('GPU'))

# 실행 중인 디바이스 확인
print("\nCurrent Divice:")
print(tf.test.gpu_device_name())

# CUDA 지원 여부
print("\nCUDA availability:", tf.test.is_built_with_cuda())

# CUDA 버전
print("\nCUDA Version", tf.__version__)