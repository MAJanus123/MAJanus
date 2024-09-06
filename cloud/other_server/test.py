switch_num = [10, 20, 3]
list_switch_num = ', '.join(map(str, switch_num))
with open(
        '/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_ablation3/experiment/' + "_switch_num.txt",
        'w') as file:
    file.write(list_switch_num)