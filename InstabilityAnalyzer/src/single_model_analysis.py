from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.bounds_propagation.new_bounds_manager import NewBoundsManager
from pynever.strategies.conversion.representation import load_network_path, ONNXNetwork
from pynever.strategies.abstraction.networks import networks
from pynever.strategies.conversion.converters.onnx import ONNXConverter
import warnings
from InstabilityAnalyzer.src.utils import *

warnings.simplefilter("error", RuntimeWarning)

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class AnalyzeModel:
    def __init__(self, model_path, dataset_loader, n_samples):
        self.model_path = model_path
        self.n_samples = n_samples
        self.n_hidden_layers = None
        self.labels = None
        self.dataset_loader =  dataset_loader


    def compute_bounds(self, noise):
        """
          This method return a matrix containing the numbers of unstable neuron per layer for each sample in the dataset
          loader. The rows represent the sample, the columns the layers (a couple for layer - hidden and upper)
          """

        if not os.path.isfile(self.model_path):
            raise Exception(f'Error: file {self.model_path} not found!')

        alt_repr = load_network_path(self.model_path)

        if not isinstance(alt_repr, ONNXNetwork):
            raise Exception('The network is not an ONNX network!')

        network = ONNXConverter().to_neural_network(alt_repr)

        if not isinstance(network, networks.SequentialNetwork):
            raise Exception('The network is not a sequential network!')

        to_ret = list()

        for index, sample in enumerate(self.dataset_loader):

            image, label = sample  # MNIST restituisce (immagine, etichetta)
            lower = image.view(-1) - noise
            upper = image.view(-1) + noise
            input = HyperRectangleBounds(lower, upper)

            results_dict = NewBoundsManager(network, input_bounds=input)
            bounds_dict, num_bounds = results_dict.propagate_bounds()
            to_ret.append(bounds_dict)

        return to_ret


    def analyze(self, bounds_list):
        to_ret = None
        for bounds_data in bounds_list:
            if self.n_hidden_layers is None:
                bounds = copy.deepcopy(bounds_data)
                self.n_hidden_layers = get_n_hidden_layers(bounds)
                to_ret = SmartDict(self.n_hidden_layers, self.n_samples)
                self.labels = get_labels(bounds)


            to_ret.update(count_unstable_neurons(bounds_data))

        return to_ret


    def analyze_neurons_activation_frequency(self, bounds_list):
        list_of_unstable_hotmaps = list()
        list_of_positive_stable_hotmaps = list()
        list_of_negative_stable_hotmaps = list()

        for bounds_data in bounds_list:
            unstable_hot_maps, positive_stable_hot_maps, negative_stable_hot_maps = get_activation_hot_map(bounds_data)
            list_of_unstable_hotmaps.append(unstable_hot_maps)
            list_of_positive_stable_hotmaps.append(positive_stable_hot_maps)
            list_of_negative_stable_hotmaps.append(negative_stable_hot_maps)


        list_of_unstable_hotmaps = list(zip(*list_of_unstable_hotmaps))
        list_of_positive_stable_hotmaps = list(zip(*list_of_positive_stable_hotmaps))
        list_of_negative_stable_hotmaps = list(zip(*list_of_negative_stable_hotmaps))

        unstable_frequency_map = list()
        stable_positive_frequency_map = list()
        stable_negative_frequency_map = list()


        for i in range(len(list_of_unstable_hotmaps)):
            i_unstable_frequency_map = torch.zeros(len(list_of_unstable_hotmaps[i][0]))
            i_stable_positive_frequency_map = torch.zeros(len(list_of_unstable_hotmaps[i][0]))
            i_stable_negative_frequency_map = torch.zeros(len(list_of_unstable_hotmaps[i][0]))

            for j in range(len(list_of_unstable_hotmaps[i])):
                i_unstable_frequency_map += list_of_unstable_hotmaps[i][j]
                i_stable_positive_frequency_map += list_of_positive_stable_hotmaps[i][j]
                i_stable_negative_frequency_map += list_of_negative_stable_hotmaps[i][j]

            unstable_frequency_map.append(i_unstable_frequency_map)
            stable_positive_frequency_map.append(i_stable_positive_frequency_map)
            stable_negative_frequency_map.append(i_stable_negative_frequency_map)


        return unstable_frequency_map, stable_positive_frequency_map, stable_negative_frequency_map






