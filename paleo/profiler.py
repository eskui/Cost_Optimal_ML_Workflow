import logging
import os

import click
import numpy as np

from paleo.graph import OperationGraph
from paleo import device
from paleo import profilers
from paleo import simulation
from paleo.utils import save_layer
from paleo import comm

#FORMAT = "%(levelname)s %(pathname)s:%(lineno)d] %(message)s"
#logging.basicConfig(format=FORMAT)
#logger = logging.getLogger("paleo")
#logger.setLevel(logging.INFO)


class Profiler():
    def __init__(self, filename, separator='\t'):
        """Initialize a profiler for the given network architecture."""
        self._filename = filename

        # Parse the net spec and flatten into a list in topology order.
        self.graph = OperationGraph(filename)
        logger.debug('Net spec loaded from %s.' % filename)
        logger.debug('Dependencies: %s' % str(self.graph.nested_list))
        self._separator = separator

    def print_static_summary(self):
        """Print a static summary about the network."""
        print('A summary of static characteristics of network.')
        print('  LAYER\tOUTPUTS')
        num_params = 0
        weights_in_bytes = 0
        num_activations = 0
        for layer_spec in self.graph.topology_order:
            layer = layer_spec.layer_op
            print('  %s' % layer)
            num_params += layer.num_params
            weights_in_bytes += layer.weights_in_bytes
            num_activations += np.prod(layer.outputs)
        print('Number of params: {:,} ({:,} Bytes)'.format(num_params,
                                                           weights_in_bytes))
        print('Activation: {:,} Bytes'.format(num_activations * 4))

    def save_conv_layers(self, save_dir):
        """Save convolution layers into separate files."""
        for layer_spec in self.graph.topology_order:
            if layer_spec['type'] != 'Convolution':
                continue
            layer = layer_spec.layer_op
            outfilename = os.path.join(save_dir, "%s.json" % layer_spec.name)
            save_layer.save_conv_layer(outfilename, layer)

    def profile(self, device_name, options, executor=None):
        """Profile the network with the given device spec.

        Returns:
            A dictionary contains the following keys:
              (layers, flops, executor, executor_std, flops_message,
              executor_msg)
        """
        device_spec = device.DEVICES[device_name]
        logger.info('Profiling for device %s' % device_spec.name)

        results = []
        for layer_spec in self.graph.topology_order:
            layer = layer_spec.layer_op

            # Always run flop-based profiler.
            if executor == 'tensorflow':
                # Here we disable the cudnn heuristics.
                # Tensorflow requires creating a cuda stream and does not allow
                # multiple context under one process.
                # We cannot use cuda stream because of the python wrapper.
                options.use_cudnn_heuristics = False

            flops_profiler = profilers.FlopsProfiler(options, device_spec)
            flop_based_time = flops_profiler.profile(layer)

            logger.info('Layer: %s' % layer_spec.name)
            logger.info('- %s: %s  %s' % (flops_profiler.name, flop_based_time,
                                          flops_profiler.message))

            if device_spec.is_gpu:
                profiler = None
                if executor == 'cudnn':
                    from profilers.cudnn_profiler import CudnnProfiler
                    profiler = CudnnProfiler(options)
                elif executor == 'tensorflow':
                    from profilers.tensorflow_profiler import (
                        TensorFlowProfiler)
                    profiler = TensorFlowProfiler(options)

                if profiler:
                    executor_time = profiler.profile(layer)
                    logger.info('- %s: %s  %s' % (profiler.name, executor_time,
                                                  profiler.message))

                    results.append(
                        (layer_spec.name, flop_based_time.total_time,
                         executor_time.total_time, 0, flops_profiler.message,
                         profiler.message))
        return results

    def profile_full_pass(self, device, num_warmup, num_iter, batch_size):
        """Profile full pass execution with tensorflow."""
        options = profilers.ProfilerOptions()
        options.num_warmup = num_warmup
        options.num_iter = num_iter
        options.include_bias_and_activation = False
        from profilers.tensorflow_profiler import TensorFlowProfiler
        profiler = TensorFlowProfiler(options)

        if batch_size:
            for l in self.graph.topology_order:
                l.layer_op.batch_size = batch_size

        layers = [
            layer_spec.layer_op for layer_spec in self.graph.topology_order
        ]

        return profiler.profile_full_pass(layers)

    def simulate(self, device_name, network_name, batch_size, use_pipeline,
                 use_only_gemm, worker_sizes, scaling, ppp_comp, ppp_comm,
                 parallel, hybrid_workers):
        device_spec = device.DEVICES[device_name]
        network_spec = device.NETWORKS[network_name]

        if parallel == 'data':
            for scaling_option in scaling.split(','):
                # Estimate time for weights update.
                # Weak scaling.
                print('=' * 10)
                headers, scaling_times = simulation.simulate_scaling(
                    self.graph.nested_list, self.graph.topology_order,
                    worker_sizes, scaling_option, batch_size, device_spec,
                    network_spec, use_pipeline, use_only_gemm, ppp_comp,
                    ppp_comm)
                print('%s scaling' % scaling_option)
                print('Profiling for device %s and %s (%f GB/s)' %
                      (device_spec.name, network_spec.name,
                       network_spec.bandwidth / 8))
                print('Use pipelining: %s' % use_pipeline)
                print('Use gemm: %s  PPP comp: %f   PPP comm: %f' %
                      (use_only_gemm, ppp_comp, ppp_comm))
                print(self._separator.join(headers))
                for times in scaling_times:
                    print(self._separator.join([str(t) for t in times]))
                # return scaling_times
        elif parallel == 'model':
            # Estimate time for weights update.
            # Weak scaling.
            print('=' * 10)
            print('Model parallel')
            headers, result_times = simulation.simulate_model_parallel(
                self.graph.nested_list, self.graph.topology_order, batch_size,
                device_spec, network_spec, use_pipeline, use_only_gemm,
                ppp_comp, ppp_comm)
            print('Profiling for device %s and %s (%f GB/s)' %
                  (device_spec.name, network_spec.name,
                   network_spec.bandwidth / 8))
            print('Use pipelining: %s' % use_pipeline)
            print('Use gemm: %s  PPP comp: %f   PPP comm: %f' %
                  (use_only_gemm, ppp_comp, ppp_comm))
            print(self._separator.join(headers))
            for times in result_times:
                print(self._separator.join([str(t) for t in times]))
        elif parallel == 'hybrid':
            # Estimate time for weights update.
            print('=' * 10)
            print('Hybrid parallel')
            headers, result_times = simulation.simulate_hybrid_parallel(
                self.graph.nested_list, self.graph.topology_order, batch_size,
                device_spec, network_spec, use_pipeline, use_only_gemm,
                ppp_comp, ppp_comm, hybrid_workers)
            print('Profiling for device %s and %s (%f GB/s)' %
                  (device_spec.name, network_spec.name,
                   network_spec.bandwidth / 8))
            print('Use pipelining: %s' % use_pipeline)
            print('Hybrid workers: %d' % hybrid_workers)
            print('Use gemm: %s  PPP comp: %f   PPP comm: %f' %
                  (use_only_gemm, ppp_comp, ppp_comm))
            print(self._separator.join(headers))
            for times in result_times:
                print(self._separator.join([str(t) for t in times]))


class BaseProfiler(object):
    """API for creating customized profilers."""

    def __init__(self, filename, device, network, separator='\t'):
        """Initialize a profiler for the given network architecture."""
        self.graph = OperationGraph(filename)
        self._options = {
            'use_only_gemm': True,
            'use_pipeline': False,
            'ppp_comp': 0.62,
            'ppp_comm': 0.72
        }
        self.device_spec = device
        self.network_spec = network

    def estimate_forward(self, batch_sizes):
        forward_times, params_in_bytes = simulation._profile_for_batch_size(
            self.graph.topology_order, 'forward', self.device_spec,
            batch_sizes, self._options['use_only_gemm'],
            self._options['ppp_comp'], self._options['ppp_comm'])

        if self._options['use_pipeline']:
            return sum([t.lowerbound for t in forward_times]), params_in_bytes

        return sum(forward_times).total_time, params_in_bytes

    def estimate_backward(self, batch_sizes):
        backward_times, _ = simulation._profile_for_batch_size(
            self.graph.topology_order, 'backward', self.device_spec,
            batch_sizes, self._options['use_only_gemm'],
            self._options['ppp_comp'], self._options['ppp_comm'])

        if self._options['use_pipeline']:
            return sum([t.lowerbound for t in backward_times])
        return sum(backward_times).total_time

    def estimate_update(self, params_in_bytes):
        time_apply_updates = simulation._profile_for_apply_updates(
            params_in_bytes, self.device_spec)
        if self._options['use_pipeline']:
            return time_apply_updates.lowerbound
        return time_apply_updates.total_time

    def estimate_comm(self, workers, params_in_bytes, scheme='TreeAllReduce'):
        comm_scheme = comm.get_comm_scheme(scheme, workers, self.network_spec,
                                           self._options['ppp_comm'])
        return comm_scheme.all_reduce(params_in_bytes)
