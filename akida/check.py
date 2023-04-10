import matplotlib.pyplot as plt
import pickle
dataset_path = 'dataset/dataset_50_50_count-False.pickle'
dataset_path = 'dataset/dataset_50_50_count-True.pickle'
# akida\dataset\dataset_50_50_count-False.pickle
with open(dataset_path, 'rb') as f:
    train_lst, label_lst = pickle.load(f)

# print(max(train_lst))
for x, y in zip(train_lst, label_lst):
    print(x.max())
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(x)
    ax2.imshow(y)
    plt.show()