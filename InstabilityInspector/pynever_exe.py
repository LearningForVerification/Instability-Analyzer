import onnx
import random
import InstabilityInspector.pynever.strategies.verification as pyn_ver
import InstabilityInspector.pynever.strategies.conversion as pyn_con
import InstabilityInspector.pynever.strategies.smt_reading as pyn_smt
import InstabilityInspector.pynever.strategies.bp.bounds_manager as bp


def py_run(network_path: str, prop_path: str, complete: bool):
    net_id = ''.join(str(random.randint(0, 9)) for _ in range(5))

    onnx_network = pyn_con.ONNXNetwork(net_id, onnx.load(network_path))
    network = pyn_con.ONNXConverter().to_neural_network(onnx_network)

    smt_parser = pyn_smt.SmtPropertyParser(prop_path, "X", "Y")
    smt_parser.parse_property()
    prop = pyn_ver.NeVerProperty(smt_parser.in_coef_mat, smt_parser.in_bias_mat, smt_parser.out_coef_mat,
                                 smt_parser.out_bias_mat)

    ver_param = [[1000] for _ in range(network.count_relu_layers())]

    if complete:
        verifier = pyn_ver.NeverVerification("best_n_neurons", ver_param)
        verifier.verify(network, prop)
        df_dict = verifier.return_df_dict()
        to_ret = df_dict
    else:
        bounds_manager = bp.BoundsManager(network, prop)
        overapprox_df_dict = bounds_manager.return_df_dict()
        to_ret = overapprox_df_dict

    return to_ret
