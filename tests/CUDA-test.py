import torch as th

print(th.__version__)
print(th.version.cuda)
print(th.cuda.is_available())
print(th.cuda.get_device_name())
print(th.cuda.get_arch_list())
print(th.zeros(1, device = 'cuda') + 1)