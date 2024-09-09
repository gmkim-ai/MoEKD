import argparse

def main():
    parser = argparse.ArgumentParser(description='compute_score')
    parser.add_argument('--name', default=None, help='path to model log.txt')
    parser.add_argument('--path', default=None, help='metric type')
    parser.add_argument('--metric', default='rougeL', help='metric type')
    args = parser.parse_args()

    args.name = args.name.replace('/', '_')
    data_list = ['dolly-512', 'self_inst-512', 'sinst_11_-512', 'uinst_11_-512', 'vicuna-512']
    seed_list = ['10', '20', '30', '40', '50']
    total_avg = 0.0
    for data in data_list:
        total_score = 0.0
        std_list = []
        for seed in seed_list:
            if args.path is not None:
                target_path = '%s/%s/%s/%s/log.txt' % (args.path, data, args.name, seed)
            else:
                target_path = 'results/moe/eval_main/%s/%s/%s/log.txt' % (data, args.name, seed)
            with open(target_path, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1]
                start_index = last_line.find("'%s': " % args.metric)
                end_index = last_line.find("}")
                score = float(last_line[start_index+len(args.metric)+4:end_index])
                total_score += score
                std_list.append(score)
        total_avg += total_score
        std_value = compute_standard_deviation(std_list)
        print("%s: %f, %f" % (data, total_score / 5.0, std_value))
    print("Total avg: %f" % (total_avg / 25.0))

def compute_standard_deviation(numbers):
    n = len(numbers)
    mean = sum(numbers) / n
    variance = sum([((x - mean) ** 2) for x in numbers]) / n
    res = variance ** 0.5
    return res
    
if __name__ == "__main__":
    main()