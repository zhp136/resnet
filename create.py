from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./data/1/train2024'],
                      test_folders=['./data/BSD100'],
                      min_size=100,
                      output_folder='./data/')