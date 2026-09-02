import numpy as np
import matplotlib.pyplot as plt
import pickle
from Simulator import  PROJECT_ROOT

# Example usage
if __name__ == "__main__":
    from Simulator.Plotter import ShapeDrawer_3D, plot_errors


    # A = np.array([
    #     [1, 0], [0, 1], [-1, 0], [0, -1],
    #     [1, 1], [-1, -1], [1, -1], [-1, 1]
    # ])
    # b = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7, 0.7]).reshape(-1,1)
    #
    # with open(f'{PROJECT_ROOT}/results/polygon/A_list.pkl', "rb") as f:
    #     A_list = pickle.load(f)
    # with open(f'{PROJECT_ROOT}/results/polygon/b_list.pkl', "rb") as f:
    #     b_list = pickle.load(f)
    # with open(f'{PROJECT_ROOT}/results/polygon/error_history.pkl', "rb") as f:
    #     error_history = pickle.load(f)
    #
    # drawer = ShapeDrawer_3D(tube_alpha=0.07, angle=(23,-106.01))
    # approximations = [(A_list[i],b_list[i].reshape(-1,1)) for i in range(0,len(A_list),3)]+[(A_list[-1],b_list[-1].reshape(-1,1))]
    # drawer.plot_polygon_evolution(A, b, approximations,save_path=f'{PROJECT_ROOT}/results/polygon/figures/pretrain_process/3d_idx.svg',show=True,title=None)
    #
    # drawer = ShapeDrawer_3D(tube_alpha=0.0, angle=(20, -112))
    # idxs = [0,3,5,8,10]
    # # idxs = [10]
    # for i in idxs:
    #     approximations = [(A_list[i], b_list[i].reshape(-1, 1))]
    #
    #     drawer.plot_polygon_evolution(A, b, approximations,save_path=f'{PROJECT_ROOT}/results/polygon/figures/pretrain_process/3d_idx_{i}.svg',show=True,title=None)

    # feas_error_mean = [np.mean(error_history['error_feas'][i]) for i in range(len(error_history['error_feas']))]
    # opt_error_mean = [np.mean(error_history['error_opt'][i]) for i in range(len(error_history['error_opt']))]
    # feas_error_max = [np.max(error_history['error_feas'][i]) for i in range(len(error_history['error_feas']))]
    # opt_error_max = [np.max(error_history['error_opt'][i]) for i in range(len(error_history['error_opt']))]
    # total_error_mean = [feas_error_mean[i]+opt_error_mean[i] for i in range(len(opt_error_mean))]
    # total_error_max = [feas_error_max[i]+opt_error_max[i] for i in range(len(opt_error_max))]
    # plot_errors(feas_error_mean,opt_error_mean,n_train=1000)
    #
    # Sigma = np.array([
    #     [5 / 2, -3 / 2],
    #     [-3 / 2, 5 / 2]
    # ])
    #
    # with open(f'{PROJECT_ROOT}/results/ellipse/A_list.pkl', "rb") as f:
    #     A_list = pickle.load(f)
    # with open(f'{PROJECT_ROOT}/results/ellipse/b_list.pkl', "rb") as f:
    #     b_list = pickle.load(f)
    # with open(f'{PROJECT_ROOT}/results/ellipse/error_history.pkl', "rb") as f:
    #     error_history = pickle.load(f)
    # # idxs = [error_history['iterations'][i]+1 for i in range(0,len(error_history['error_feas']),2)]+[error_history['iterations'][-1]+1]
    # approximations = [(A_list[i],b_list[i].reshape(-1,1)) for i in range(0,len(A_list),3)]+[(A_list[-1],b_list[-1].reshape(-1,1))]
    # # drawer.plot_ellipse_evolution(Sigma, approximations)
    #
    # drawer = ShapeDrawer_3D(tube_alpha=0.0, angle=(0, -114.))
    # idxs = [0,3,5,8,10]
    # # idxs = [10]
    # for i in idxs:
    #     approximations = [(A_list[i], b_list[i].reshape(-1, 1))]
    #     drawer.plot_ellipse_evolution(Sigma, approximations,save_path=f'{PROJECT_ROOT}/results/ellipse/figures/pretrain_process/3d_idx_{i}.svg',show=True,title=None)
    #
    # feas_error_mean = [np.mean(error_history['error_feas'][i]) for i in range(len(error_history['error_feas']))]
    # opt_error_mean = [np.mean(error_history['error_opt'][i]) for i in range(len(error_history['error_opt']))]
    # feas_error_max = [np.max(error_history['error_feas'][i]) for i in range(len(error_history['error_feas']))]
    # opt_error_max = [np.max(error_history['error_opt'][i]) for i in range(len(error_history['error_opt']))]
    # total_error_mean = [feas_error_mean[i]+opt_error_mean[i] for i in range(len(opt_error_mean))]
    # total_error_max = [feas_error_max[i]+opt_error_max[i] for i in range(len(opt_error_max))]
    #
    # plot_errors(feas_error_mean,opt_error_mean,n_train=300, interporation=True)


    center1 = (0.0, 0.0)
    radius1 = 1.0
    center2 = (1.0, 1.0)
    radius2 = 1.0

    with open(f'{PROJECT_ROOT}/results/nonconvex/A_list.pkl', "rb") as f:
        A_list = pickle.load(f)
    with open(f'{PROJECT_ROOT}/results/nonconvex/b_list.pkl', "rb") as f:
        b_list = pickle.load(f)
    with open(f'{PROJECT_ROOT}/results/nonconvex/error_history.pkl', "rb") as f:
        error_history = pickle.load(f)
    # idxs = [error_history['iterations'][i]+1 for i in range(0,len(error_history['error_feas']),2)]+[error_history['iterations'][-1]+1]
    approximations = [(A_list[i],b_list[i].reshape(-1,1)) for i in range(0,len(A_list),3)]+[(A_list[-1],b_list[-1].reshape(-1,1))]
    # drawer.plot_circle_region_evolution(center1=center1, radius1=radius1, center2=center2, radius2=radius2,
    #                                     approximations=approximations)

    drawer = ShapeDrawer_3D(tube_alpha=0.0, angle=(-4, -108))
    idxs = [0,3,5,8,10]
    # idxs = [10]
    for i in idxs:
        approximations = [(A_list[i], b_list[i].reshape(-1, 1))]
        drawer.plot_circle_region_evolution(center1=center1, radius1=radius1, center2=center2,radius2=radius2, approximations=approximations,save_path=f'{PROJECT_ROOT}/results/nonconvex/figures/pretrain_process/3d_idx_{i}.svg',show=True,title=None)
    #
    # feas_error_mean = [np.mean(error_history['error_feas'][i]) for i in range(len(error_history['error_feas']))]
    # opt_error_mean = [np.mean(error_history['error_opt'][i]) for i in range(len(error_history['error_opt']))]
    # feas_error_max = [np.max(error_history['error_feas'][i]) for i in range(len(error_history['error_feas']))]
    # opt_error_max = [np.max(error_history['error_opt'][i]) for i in range(len(error_history['error_opt']))]
    # total_error_mean = [feas_error_mean[i]+opt_error_mean[i] for i in range(len(opt_error_mean))]
    # total_error_max = [feas_error_max[i]+opt_error_max[i] for i in range(len(opt_error_max))]
    # plot_errors(feas_error_mean,opt_error_mean,n_train=300, interporation=True)