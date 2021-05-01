from src import utils
from MCFS import MCFS
from datetime import datetime
import time


def main():
    data_set_list = ['MNIST', 'lung_small', 'warpPIE10P', 'Yale', 'digits']
    n_clusters_list = [10, 7, 10, 15, 10]
    meth = ['MCFS-I']


    for i in range(len(data_set_list)):

        print('dataset: {}'.format(data_set_list[i]))
        data, label = utils.read_data(data_set_list[i])

        # test n_neighbor
        for n_neighbor in range(1, 21):

            print('n_neighbor: {}/{}'.format(n_neighbor, 20))
            with open("../Result/" + data_set_list[i] + "_neighbor_test.txt", 'a') as f:
                line = str(n_neighbor) + '\n'
                f.write(line)

            for j in range(len(meth)):
                print('method: {}\ttime: {}'.format(meth[j], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

                for o in range(3):
                    print('step: {}/3\ttime: {}'.format(str(o+1), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    star_time = datetime.now()
                    weight = MCFS.mcfs(X=data, n_selected_features=100, i=1,
                                       n_emb=n_clusters_list[i], n_neighbors=n_neighbor)
                    idx = MCFS.feature_ranking(weight)

                    selected_data = data[:, idx[0:100]]
                    end_time = datetime.now()

                    # perform kmeans clustering based on the selected features and repeats 5 times
                    nmi_total = 0.0
                    for k in range(3):
                        nmi_total += MCFS.eval_cluster_prediction(selected_data, label, n_clusters_list[i])

                    # output the average NMI
                    with open("../Result/" + data_set_list[i] + "_neighbor_test.txt", 'a') as f:
                        line = meth[j] + ': ' + str(float(nmi_total) / 3) + '\tcost_time: ' + \
                               str((end_time-star_time).seconds) + 's\n'
                        f.write(line)

            with open("../Result/" + data_set_list[i] + "_neighbor_test.txt", 'a') as f:
                line = '\n\n'
                f.write(line)

if __name__ == '__main__':
    main()