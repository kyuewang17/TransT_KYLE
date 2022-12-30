from pytracking.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/mnt/77A934E328EE77B0/tracking/GOT-10k/'
    settings.got_packed_results_path = '/home/kyle/PycharmProjects/TransT_KYLE/pytracking/tracking_results/got_packed_results/'
    settings.got_reports_path = ''
    settings.lasot_path = '/mnt/77A934E328EE77B0/tracking/LaSOT/'
    settings.network_path = '/home/kyle/PycharmProjects/TransT_KYLE/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/mnt/77A934E328EE77B0/tracking/OTB/OTB2015'
    settings.result_plot_path = '/home/kyle/PycharmProjects/TransT_KYLE/pytracking/result_plots/'
    settings.results_path = '/home/kyle/PycharmProjects/TransT_KYLE/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/kyle/PycharmProjects/TransT_KYLE/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/16E622223B4BD641/TrackingNet'
    settings.uav_path = '/mnt/77A934E328EE77B0/tracking/UAV123/'
    settings.vot_path = '/mnt/77A934E328EE77B0/VOT2018'
    settings.youtubevos_dir = ''

    return settings

