def process_args(args):
    args.no_known_index = len(args.chars) # no_know char
    args.pad_rec_index = args.no_known_index + 1 # pad_rec char

    args.end_index = args.pad_rec_index + 1 # EOS
    args.start_index = args.end_index + 1 # SOS
    args.noise_index = args.start_index + 1 # noise char

    print('len(args.chars)',len(args.chars))
    print('no_known_index',args.no_known_index)
    print('pad_rec_index',args.pad_rec_index)
    print('end_index',args.end_index)
    print('start_index',args.start_index)
    print('noise_index',args.noise_index)

    return args

# 最后的字典组成元素为：
# chars 0~94
# no_known_index 95
# pad_rec_index 96
# end_index 97
# start_index 98
# noise_index 99

