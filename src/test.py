from src import utils
from cmp import lap_score
from cmp import NDFS
from cmp.sparse_learning import feature_ranking
from cmp.construct_W import construct_W
from MCFS import MCFS
from datetime import datetime
import time


def main():
    data_set_list = ['MNIST', 'lung_small', 'warpPIE10P', 'Yale', 'digits']
    n_clusters_list = [10, 7, 10, 15, 10]
    n_select_feature = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    num_select_feature_max = [200, 200, 200, 200, 64]
    meth = ['MCFS', 'MCFS-I', 'lap_score', 'NDFS']


    for i in range(len(data_set_list)):
        """
            i = 0: use data set MNIST
            i = 1: use data set lung_small
            i = 2: use data set warpPIE10P
            i = 3: use data set Yale
            i = 4: use data set digits
        """

        print('dataset: {}'.format(data_set_list[i]))
        data, label = utils.read_data(data_set_list[i])
        for num_sel_fea in n_select_feature:
            if num_sel_fea > num_select_feature_max[i]:
                break

            print('select feature: {}/{}'.format(num_sel_fea, num_select_feature_max[i]))
            with open("../Result/" + data_set_list[i] + ".txt", 'a') as f:
                line = str(num_sel_fea) + '\n'
                f.write(line)

            for j in range(len(meth)):
                """
                    j = 0: test MCFS
                    j = 1: test MCFS-I
                    j = 2: test lap_score
                    j = 3: test NDFS
                """
                print('method: {}\ttime: {}'.format(meth[j], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

                cnt = 0
                for o in range(5):
                    print('step: {}/5\ttime: {}'.format(str(o+1), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    star_time = datetime.now()
                    if j < 2:
                        weight = MCFS.mcfs(X=data, n_selected_features=num_sel_fea, i=j,
                                           n_emb=n_clusters_list[i], n_neighbors=5)
                        idx = MCFS.feature_ranking(weight)
                    elif j == 2:
                        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn",
                                    "weight_mode": "heat_kernel", "k": 5, 't': 1}
                        W = construct_W(data, **kwargs_W)
                        score = lap_score.lap_score(data, W=W)
                        idx = lap_score.feature_ranking(score)
                    elif j == 3:
                        kwargs = {"metric": "euclidean", "neighborMode": "knn",
                                  "weightMode": "heatKernel", "k": 5, 't': 1}
                        W = construct_W(data, **kwargs)
                        Weight = NDFS.ndfs(data, W=W, n_clusters=20)
                        idx = feature_ranking(Weight)

                    selected_data = data[:, idx[0:num_sel_fea]]
                    end_time = datetime.now()
                    print((end_time-star_time).microseconds)
                    cnt += (end_time-star_time).microseconds

                    # perform k-means clustering based on the selected features and repeats 5 times
                    nmi_total = 0.0
                    for k in range(5):
                        nmi_total += MCFS.eval_cluster_prediction(selected_data, label, n_clusters_list[i])

                    # output the average NMI
                    with open("../Result/" + data_set_list[i] + ".txt", 'a') as f:
                        line = meth[j] + ': ' + str(float(nmi_total) / 5) + '\tcost_time: ' + \
                               str((end_time-star_time).microseconds) + 'us\n'
                        f.write(line)

            with open("../Result/" + 'MCFS-I' + ".txt", 'a') as f:
                line = '\n\n'
                f.write(line)

    print('Test is complete!')
    utils.send_message('Complete', 'Test is complete!')

if __name__ == '__main__':
    main()