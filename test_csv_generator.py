import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from InstabilityInspector.Inspector import Inspector
from InstabilityInspector.instability_analysis import Analyzer
import torchvision.transforms as tr

DATASET = "fashion_mnist"
DATASET_DIR = "dataset"

if __name__ == '__main__':
    # # generate dataset
    # (ds_train, ds_test), ds_info = tfds.load(
    #     DATASET,
    #     split=['train', 'test'],
    #     shuffle_files=True,
    #     as_supervised=True,
    #     with_info=True,
    # )
    #
    # ds_train = ds_train.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # ds_train = ds_train.batch(128)
    # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    #
    # ds_test = ds_test.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.batch(128)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Normalize((0.1307,), (0.3081,)),
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])

    train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

    inspector = Inspector(r"/Users/andrea/Desktop/Instability-Analizer/experiments/model1.onnx", "experiments",
                          test_dataset)
    dict = inspector.bounds_inspector(5, 0.05, 0.15, False, True)

    analysis = Analyzer(inspector.get_output_folder(),
                                   r"/Users/andrea/Desktop/Instability-Analizer/experiments/output")
    analysis.analyze()
