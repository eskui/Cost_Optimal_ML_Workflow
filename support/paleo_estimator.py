import paleo.profiler as profilers
import paleo.device as device

"""
device.Device args:
    name: device name
    clock: MHz
    peek_gflop: GFLOPS
    mem_bandwidth: GB/sec
"""

NETWORK = device.Network("EC2_network", 20)
DEVICE_CPU = device.Device("C5D_CPU",3000,1165,119.21,False)
DEVICE_A10GPU = device.Device("G5_A10GPU",885,31240,600,True)
DEVICE_M60GPU = device.Device("G3S_M60GPU",899,9650,320,True)

def estimate_required_time(device_name):
    if device_name == "G5_A10GPU":
        device = DEVICE_A10GPU
    elif device_name == "G3S_M60GPU":
        device = DEVICE_M60GPU
    elif device_name == "C5D_CPU":
        device = DEVICE_CPU
    else:
        print("Please give a proper device and try again")
        exit()

    profiler = profilers.BaseProfiler("conv_network.json", device, NETWORK)
    forward_time, kbytes = profiler.estimate_forward(32)
    backward_time = profiler.estimate_backward(32)
    update_time = profiler.estimate_update(kbytes)
    total_time_ms = (forward_time + backward_time + update_time)
    total_time_seconds = total_time_ms*0.001
    print("Estimated one batch gradient updated time: {} seconds".format(total_time_seconds))
    return total_time_ms
