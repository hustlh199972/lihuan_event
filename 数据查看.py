import numpy as np
a=np.load('D:\ecoco_depthmaps_test\ecoco_depthmaps_test/train\sequence_0000000000\VoxelGrid-betweenframes-5\event_tensor_0000000000.npy')
b=np.load('D:\ecoco_depthmaps_test\ecoco_depthmaps_test/train\sequence_0000000000/flow\disp01_0000000001.npy')
print(a.shape,b.shape)