

def datasets_path(dataset_name='', task_name='', speeds='4000r', ele_frequency='1600Hz', vib_frequency='1600Hz'):
    if dataset_name == 'PU_datasets_3classes_4096_1samples':
        class_numbers = 3  # 故障类别数量
        data_root = r'./dataset/PU_datasets_3classes_4096_1samples/'  # PU_datasets_3classes_4096_1samples
        dataset_name = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
        if task_name == 'T1':
            source_dataset_path_list = ['PU_N15_M01_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            # source_dataset_path_list = [data_root + '_data.npy' for name in source_dataset_path_list]
            target_dataset_path = 'PU_N09_M07_F10'
        elif task_name == 'T2':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M01_F10'
        elif task_name == 'T3':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M07_F04'
        elif task_name == 'T0':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F04']
            target_dataset_path = 'PU_N15_M07_F10'
    elif dataset_name == 'PU_datasets_4classes_4096_1samples':
        class_numbers = 4
        data_root = r'./dataset/PU_datasets_4classes_4096_1samples/'  # PU_datasets_4classes_4096_1samples  PU_datasets_4classes_4096_1samples_norm
        if task_name == 'T1':
            source_dataset_path_list = ['PU_N15_M01_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N09_M07_F10'
        elif task_name == 'T2':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M01_F10'
        elif task_name == 'T3':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M07_F04'
        elif task_name == 'T0':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F04']
            target_dataset_path = 'PU_N15_M07_F10'
    elif dataset_name == 'HUST_motor_datasets':
        class_numbers = 6
        data_root = r'./dataset/HUST_motor/'  # HUST_motor_2048  HUST_motor_1024_norm
        if task_name == 'T5':
            source_dataset_path_list = ['HUST_10Hz_6', 'HUST_20Hz_6', 'HUST_30Hz_6']
            target_dataset_path = 'HUST_05Hz_6'
        elif task_name == 'T10':
            source_dataset_path_list = ['HUST_05Hz_6', 'HUST_20Hz_6', 'HUST_30Hz_6']
            target_dataset_path = 'HUST_10Hz_6'
        elif task_name == 'T20':
            source_dataset_path_list = ['HUST_05Hz_6', 'HUST_10Hz_6', 'HUST_30Hz_6']
            target_dataset_path = 'HUST_20Hz_6'
        elif task_name == 'T30':
            source_dataset_path_list = ['HUST_05Hz_6', 'HUST_10Hz_6', 'HUST_20Hz_6']
            target_dataset_path = 'HUST_30Hz_6'
    elif dataset_name == 'BJUT_WT_datasets':
        class_numbers = 5
        data_root = r'./dataset/BJUT_WT_4096_64samples/'  # BJUT_WT_4096_64samples_norm
        if task_name == 'T20':
            source_dataset_path_list = ['BJUT_30Hz_5', 'BJUT_40Hz_5', 'BJUT_50Hz_5']
            target_dataset_path = 'BJUT_20Hz_5'
        elif task_name == 'T30':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_40Hz_5', 'BJUT_50Hz_5']
            target_dataset_path = 'BJUT_30Hz_5'
        elif task_name == 'T40':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_30Hz_5', 'BJUT_50Hz_5']
            target_dataset_path = 'BJUT_40Hz_5'
        elif task_name == 'T50':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_30Hz_5', 'BJUT_40Hz_5']
            target_dataset_path = 'BJUT_50Hz_5'
    elif dataset_name == 'BJUT_WT_datasets_IFAC':
        class_numbers = 5
        data_root = r'./dataset/BJUT_WT_64_samples_all_speed/'  # BJUT_WT_4096_64samples_norm
        if task_name == 'T20':
            source_dataset_path_list = ['BJUT_25Hz_5', 'BJUT_30Hz_5', 'BJUT_35Hz_5']
            target_dataset_path = 'BJUT_20Hz_5'
        elif task_name == 'T25':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_30Hz_5', 'BJUT_35Hz_5']
            target_dataset_path = 'BJUT_25Hz_5'
        elif task_name == 'T30':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_25Hz_5', 'BJUT_35Hz_5']
            target_dataset_path = 'BJUT_30Hz_5'
        elif task_name == 'T35':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_25Hz_5', 'BJUT_30Hz_5']
            target_dataset_path = 'BJUT_35Hz_5'

    else:
        class_numbers = 4
        data_root = r'./dataset/SCARA_datasets/electrical_vibration_{}_vib/第1批/{}/{}_elc/'.format(vib_frequency, speeds, ele_frequency)
        if task_name == 'T0':
            source_dataset_path_list = ['SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)
        elif task_name == 'T3':
            source_dataset_path_list = ['SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)
        elif task_name == 'T6':
            source_dataset_path_list = ['SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)
        elif task_name == 'T9':
            source_dataset_path_list = ['SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)

    return data_root, source_dataset_path_list[0], source_dataset_path_list[1], source_dataset_path_list[2], \
        target_dataset_path, class_numbers







def datasets_path_0(dataset_name='', task_name='', speeds='4000r', ele_frequency='1600Hz', vib_frequency='1600Hz'):
    # print("ccccccccccccccccccccccccccccccccccc              dataset_name:", dataset_name)
    if dataset_name == 'PU_datasets_3classes_4096_1samples':
        class_numbers = 3  # 故障类别数量
        data_root = r'.\dataset\PU_datasets_3classes_4096_1samples\\'  # PU_datasets_3classes_4096_1samples  PU_datasets_4classes_4096_1samples_norm
        dataset_name = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
        if task_name == 'T1':
            source_dataset_path_list = ['PU_N15_M01_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            # source_dataset_path_list = [data_root + '_data.npy' for name in source_dataset_path_list]
            target_dataset_path = 'PU_N09_M07_F10'
        elif task_name == 'T2':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M01_F10'
        elif task_name == 'T3':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M07_F04'
        elif task_name == 'T0':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F04']
            target_dataset_path = 'PU_N15_M07_F10'
    elif dataset_name == 'PU_datasets_4classes_4096_1samples':  # PU_datasets_4classes_4096_1samples PU_datasets_4classes_4096_1samples_norm
        class_numbers = 4
        data_root = r'.\dataset\PU_datasets_4classes_4096_1samples\\'
        if task_name == 'T1':
            source_dataset_path_list = ['PU_N15_M01_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N09_M07_F10'
        elif task_name == 'T2':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M07_F04', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M01_F10'
        elif task_name == 'T3':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F10']
            target_dataset_path = 'PU_N15_M07_F04'
        elif task_name == 'T0':
            source_dataset_path_list = ['PU_N09_M07_F10', 'PU_N15_M01_F10', 'PU_N15_M07_F04']
            target_dataset_path = 'PU_N15_M07_F10'
    elif dataset_name == 'SUDA_dataset':
        class_numbers = 4
        data_root = r'.\dataset\SCARA_datasets\第1批\4000r\1600Hz_elc\\'
        dataset_name = ['SCARA_0kg_4000r_1600Hz_elc', 'SCARA_3kg_4000r_1600Hz_elc',
                        'SCARA_6kg_4000r_1600Hz_elc', 'SCARA_9kg_4000r_1600Hz_elc']
        if task_name == 'T0':
            source_dataset_path_list = ['SCARA_3kg_4000r_1600Hz_elc', 'SCARA_6kg_4000r_1600Hz_elc',
                                        'SCARA_9kg_4000r_1600Hz_elc']
            target_dataset_path = 'SCARA_0kg_4000r_1600Hz_elc'
        elif task_name == 'T3':
            source_dataset_path_list = ['SCARA_0kg_4000r_1600Hz_elc', 'SCARA_6kg_4000r_1600Hz_elc',
                                        'SCARA_9kg_4000r_1600Hz_elc']
            target_dataset_path = 'SC ARA_3kg_4000r_1600Hz_elc'
        elif task_name == 'T6':
            source_dataset_path_list = ['SCARA_0kg_4000r_1600Hz_elc', 'SCARA_3kg_4000r_1600Hz_elc',
                                        'SCARA_9kg_4000r_1600Hz_elc']
            target_dataset_path = ['SCARA_6kg_4000r_1600Hz_elc']
        elif task_name == 'T9':
            source_dataset_path_list = ['SCARA_0kg_4000r_1600Hz_elc', 'SCARA_3kg_4000r_1600Hz_elc',
                                        'SCARA_6kg_4000r_1600Hz_elc']
            target_dataset_path = ['SCARA_9kg_4000r_1600Hz_elc']
    elif dataset_name == 'BJUT_datasets':
        class_numbers = 5
        data_root = r'.\dataset\BJUT_WT_64_samples\\'  # BJUT_WT 200个样本  BJUT_WT_100_samples  BJUT_WT_64_samples
        if task_name == 'T20':
            source_dataset_path_list = ['WT_1_30hz', 'WT_1_40hz', 'WT_1_50hz']
            target_dataset_path = 'WT_1_20hz'
        elif task_name == 'T30':
            source_dataset_path_list = ['WT_1_20hz', 'WT_1_40hz', 'WT_1_50hz']
            target_dataset_path = 'WT_1_30hz'
        elif task_name == 'T40':
            source_dataset_path_list = ['WT_1_20hz', 'WT_1_30hz', 'WT_1_50hz']
            target_dataset_path = 'WT_1_40hz'
        elif task_name == 'T50':
            source_dataset_path_list = ['WT_1_20hz', 'WT_1_30hz', 'WT_1_40hz']
            target_dataset_path = 'WT_1_50hz'

        # if task_name == 'T25':
        #     source_dataset_path_list = ['WT_1_35hz', 'WT_1_45hz', 'WT_1_55hz']
        #     target_dataset_path = 'WT_1_25hz'
        # elif task_name == 'T35':
        #     source_dataset_path_list = ['WT_1_25hz', 'WT_1_45hz', 'WT_1_55hz']
        #     target_dataset_path = 'WT_1_35hz'
        # elif task_name == 'T45':
        #     source_dataset_path_list = ['WT_1_25hz', 'WT_1_35hz', 'WT_1_55hz']
        #     target_dataset_path = 'WT_1_45hz'
        # elif task_name == 'T55':
        #     source_dataset_path_list = ['WT_1_25hz', 'WT_1_35hz', 'WT_1_45hz']
        #     target_dataset_path = 'WT_1_55hz'

        # if task_name == 'T20':
        #     source_dataset_path_list = ['WT_1_25hz', 'WT_1_30hz', 'WT_1_35hz']
        #     target_dataset_path = 'WT_1_20hz'
        # elif task_name == 'T25':
        #     source_dataset_path_list = ['WT_1_20hz', 'WT_1_30hz', 'WT_1_35hz']
        #     target_dataset_path = 'WT_1_25hz'
        # elif task_name == 'T30':
        #     source_dataset_path_list = ['WT_1_20hz', 'WT_1_25hz', 'WT_1_35hz']
        #     target_dataset_path = 'WT_1_30hz'
        # elif task_name == 'T35':
        #     source_dataset_path_list = ['WT_1_20hz', 'WT_1_25hz', 'WT_1_30hz']
        #     target_dataset_path = 'WT_1_35hz'
    elif dataset_name == 'HUST_motor_datasets':
        class_numbers = 6
        data_root = r'.\dataset\HUST_motor_1024_norm\\'  # HUST_motor_2048  HUST_motor_1024_norm
        if task_name == 'T5':
            source_dataset_path_list = ['HUST_10Hz_6', 'HUST_20Hz_6', 'HUST_30Hz_6']
            target_dataset_path = 'HUST_05Hz_6'
        elif task_name == 'T10':
            source_dataset_path_list = ['HUST_05Hz_6', 'HUST_20Hz_6', 'HUST_30Hz_6']
            target_dataset_path = 'HUST_10Hz_6'
        elif task_name == 'T20':
            source_dataset_path_list = ['HUST_05Hz_6', 'HUST_10Hz_6', 'HUST_30Hz_6']
            target_dataset_path = 'HUST_20Hz_6'
        elif task_name == 'T30':
            source_dataset_path_list = ['HUST_05Hz_6', 'HUST_10Hz_6', 'HUST_20Hz_6']
            target_dataset_path = 'HUST_30Hz_6'
    elif dataset_name == 'BJUT_WT_datasets':
        class_numbers = 5
        data_root = r'.\dataset\BJUT_WT_4096_64samples_norm\\'
        if task_name == 'T20':
            source_dataset_path_list = ['BJUT_30Hz_5', 'BJUT_40Hz_5', 'BJUT_50Hz_5']
            target_dataset_path = 'BJUT_20Hz_5'
        elif task_name == 'T30':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_40Hz_5', 'BJUT_50Hz_5']
            target_dataset_path = 'BJUT_30Hz_5'
        elif task_name == 'T40':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_30Hz_5', 'BJUT_50Hz_5']
            target_dataset_path = 'BJUT_40Hz_5'
        elif task_name == 'T50':
            source_dataset_path_list = ['BJUT_20Hz_5', 'BJUT_30Hz_5', 'BJUT_40Hz_5']
            target_dataset_path = 'BJUT_50Hz_5'

    else:
        class_numbers = 4
        data_root = r'.\dataset\SCARA_datasets\electrical_vibration_{}_vib\第1批\{}\{}_elc\\'.format(vib_frequency, speeds, ele_frequency)
        if task_name == 'T0':
            source_dataset_path_list = ['SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)
        elif task_name == 'T3':
            source_dataset_path_list = ['SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)
        elif task_name == 'T6':
            source_dataset_path_list = ['SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)
        elif task_name == 'T9':
            source_dataset_path_list = ['SCARA_0kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_3kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency),
                                        'SCARA_6kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)]
            target_dataset_path = 'SCARA_9kg_{}_e{}_v{}'.format(speeds, ele_frequency, vib_frequency)

    return data_root, source_dataset_path_list[0], source_dataset_path_list[1], source_dataset_path_list[2], \
        target_dataset_path, class_numbers
