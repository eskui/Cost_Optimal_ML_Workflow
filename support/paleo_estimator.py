import paleo.profiler as profilers
import paleo.device as device

NETWORK = device.Network("EC2_network", 20)
DEVICE_T4GPU = device.Device("NVIDIA_T4",585,8100,320,True)
DEVICE_LOCAL = device.Device("local_CPU",2300,40,2133,False)

def estimate_required_time(device_name):
    if device_name == "NVIDIA_T4":
        device = DEVICE_T4GPU
    elif device_name == "local_CPU":
        device = DEVICE_LOCAL
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
