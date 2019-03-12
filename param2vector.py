# -*- coding: utf-8 -*-

def param2vector(fileaddress = '') :
    with open(fileaddress, 'r', encoding='utf-8') as fp :
        case_list, label_list, dataset_list = [], [], []

        for line in fp.readlines() :
            if (line[0] != '{') :
                continue
            else :
                case_string = line[1:-2]
                tmp_list = list(case_string.split(',', case_string.count(',')))
                for i in range(0, len(tmp_list)) :
                    tmp_list[i] = int(tmp_list[i])
                case_list.append([1] + tmp_list[:-1])
                label_list.append(tmp_list[-1])
        dataset_list = [label_list, case_list]
        return dataset_list


if __name__ == '__main__':
    dataset_list = param2vector('./testcase.param')
    print('case_len:', len(dataset_list[1][0]))
    for i in range(0, len(dataset_list[0])) :
        print('ID:', i, 'label:', dataset_list[0][i], 'case:', dataset_list[1][i])
