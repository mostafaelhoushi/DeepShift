import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info

def summary_string(model, input_size, batch_size=-1, device="cuda", dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ""
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            params_bits = 0
            # TODO: handle batchnorm params
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                weight_params = torch.prod(torch.LongTensor(list(module.weight.size()))) 
                params += weight_params
                params_bits += weight_params * 32

                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "shift") and hasattr(module.shift, "size"):
                assert(hasattr(module, "sign"))
                assert(hasattr(module.sign, "size"))
                assert(module.shift.size() == module.sign.size())

                shift_params = torch.prod(torch.LongTensor(list(module.shift.size())))
                params += shift_params
                params_bits += shift_params * 5

                summary[m_key]["trainable"] = module.shift.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                bias_params = torch.prod(torch.LongTensor(list(module.bias.size())))
                params += bias_params
                params_bits += bias_params * 32
            if hasattr(module, "running_mean") and hasattr(module.running_mean, "size") and hasattr(module, "track_running_stats") and module.track_running_stats:
                running_mean_params = torch.prod(torch.LongTensor(list(module.running_mean.size())))
                params += running_mean_params
                params_bits += running_mean_params * 32
            if hasattr(module, "running_var") and hasattr(module.running_var, "size") and hasattr(module, "track_running_stats") and module.track_running_stats:
                running_var_params = torch.prod(torch.LongTensor(list(module.running_var.size()))) 
                params += running_var_params
                params_bits += running_var_params * 32
            summary[m_key]["nb_params"] = params
            summary[m_key]["bits_params"] = params_bits

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of at least 2 for each GPU for batchnorm
    n_samples = (torch.cuda.device_count() + 1)*2
    x = [torch.rand(n_samples, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model.eval()
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_params_bits = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_params_bits += summary[layer]["bits_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params_bits.numpy() / (8. * (1024 ** 2.)))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    
    return summary_str, (total_params, trainable_params)
