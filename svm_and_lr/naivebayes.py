import jieba
import math

division = '> * <'

ad_pr = {}
non_pr = {}

stop_word = ['0', '1', '  ']

pr_ad_prior = 0.0
pr_non_prior = 0.0

pr_ad_avg = 0.0
pr_non_avg = 0.0


def write_param(filename):
    global pr_ad_prior
    global pr_non_prior

    ad_size = 0
    non_size = 0

    ad_tf = {}
    non_tf = {}
    term_set = set()

    ad_term_size = 0
    ad_all_term_size = 0
    non_term_size = 0
    non_all_term_size = 0

    file = open(filename, encoding='utf-8')
    result_file = open(filename.split('.')[0] + '-result.txt', mode='w', encoding='utf-8')
    for line in file:
        if len(line) < 3:
            continue
        if line[0] == '1':
            ad_size += 1
        else:
            non_size += 1
        r = jieba.lcut(line)
        # r = line[2:]
        for i in range(2, len(r)):
            if r[i] not in term_set:
                term_set.add(r[i])
            if line[0] == '1':
                if r[i] not in ad_tf:
                    ad_tf[r[i]] = 1
                    ad_term_size += 1
                else:
                    ad_tf[r[i]] = ad_tf[r[i]] + 1
                ad_all_term_size += 1
            else:
                if r[i] not in non_tf:
                    non_tf[r[i]] = 1
                    non_term_size += 1
                else:
                    non_tf[r[i]] = non_tf[r[i]] + 1
                non_all_term_size += 1

    for key in ad_tf:
        ad_pr[key] = (ad_tf[key] + 1) / (ad_all_term_size + ad_term_size)
    for key in non_tf:
        non_pr[key] = (non_tf[key] + 1) / (non_all_term_size + non_term_size)

    result_file.write(str(ad_size) + ' ' + str(non_size))
    result_file.write('\n')
    result_file.write(str(1 / ad_term_size) + ' ' + str(1 / non_term_size))
    result_file.write('\n')
    for t in term_set:
        if t not in ad_pr:
            pr_ad = 1 / ad_term_size
        else:
            pr_ad = ad_pr[t]

        if t not in non_pr:
            pr_non = 1 / non_term_size
        else:
            pr_non = non_pr[t]
        result_file.write(t + division + str(pr_ad) + division + str(pr_non))
        result_file.write('\n')
    result_file.close()
    file.close()


def read_param(filename):
    global pr_ad_avg
    global pr_non_avg
    global pr_ad_prior
    global pr_non_prior
    file = open(filename, encoding='utf-8')
    i = 0
    for line in file:
        if len(line) < 3:
            continue
        if i == 0:
            ad_size = int(line.split(' ')[0])
            non_size = int(line.split(' ')[1])
        elif i == 1:
            pr_ad_avg = float(line.split(' ')[0])
            pr_non_avg = float(line.split(' ')[1])
        else:
            datas = line.split(division)
            ad_pr[datas[0]] = float(datas[1])
            non_pr[datas[0]] = float(datas[2])
        i += 1
    pr_ad_prior = ad_size / (ad_size + non_size)
    pr_non_prior = non_size / (ad_size + non_size)


def __confidence_compute(data):
    global pr_ad_prior
    global pr_non_prior

    r = jieba.lcut(data)
    # r = data
    ad_score = 0
    non_score = 0

    score = 1
    for i in range(2, len(r)):
        term = r[i]
        if term in ad_pr:
            ad_score = ad_pr[term]
        else:
            ad_score = pr_ad_avg
        if term in non_pr:
            non_score = non_pr[term]
        else:
            non_score = pr_non_avg
        score *= (ad_score / non_score)

    score *= (pr_ad_prior / pr_non_prior)
    # ad_score += math.log1p(pr_ad_prior)
    # non_score += math.log1p(pr_non_prior)
    return score


def get_result(msg):
    global pr_ad_prior
    global pr_non_prior

    if ad_pr == None or len(ad_pr) == 0:
        read_param('train-result.txt')

    r = jieba.lcut(msg)
    # r = data
    ad_score = 0
    non_score = 0

    score = 1
    for i in range(2, len(r)):
        term = r[i]
        if term in ad_pr:
            ad_score = ad_pr[term]
        else:
            ad_score = pr_ad_avg
        if term in non_pr:
            non_score = non_pr[term]
        else:
            non_score = pr_non_avg
        score *= (ad_score / non_score)

    score *= (pr_ad_prior / pr_non_prior)

    if score > 10000000:
        className = 1
    else:
        className = 0

    return 'naive_bayes', 0.980978, className, 1 / (1 + math.exp(10000000 - score))


def predict(filename):
    file = open(filename, encoding='utf-8')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    output_file = open('test_result.txt', mode='w', encoding='utf-8')

    for line in file:
        if len(line) < 3:
            continue
        confidence = __confidence_compute(line)
        output_file.write(str(confidence) + ' ' + line)
        if line[0] == '1':
            if confidence > 10000000:
                tp += 1
            else:
                print('1 ' + str(confidence))
                fn += 1
        else:
            if confidence < 10000000:
                tn += 1
            else:
                print('0 ' + str(confidence))
                fp += 1
    output_file.close()
    print('tp:%d    fp:%d\nfn:%d    tn:%d' % (tp, fp, fn, tn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('recall:%f    precision:%f\nf1:%f' % (recall, precision, recall * precision * 2 / (recall + precision)))


if __name__ == '__main__':
    # write_param('train.txt')
    # read_param('train-result.txt')
    # print(__confidence_compute('您好！我是福州融汇温泉城的高级置业顾问  彭磊，近期我们项目有做些活动，且价位非常优惠，接待点地址：福州市晋安区桂湖。也希望您继续关注'))
    print(get_result('您好！我是福州融汇温泉城的高级置业顾问  彭磊，近期我们项目有做些活动，且价位非常优惠，接待点地址：福州市晋安区桂湖。也希望您继续关注'))
    # predict('test.txt')
