from classifiers.network import local_train
import data_provider

if __name__ == '__main__':
    data_provider.init_data_provider()
    local_train()
