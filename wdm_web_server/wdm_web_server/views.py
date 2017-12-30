from django.http import HttpResponse
import json
import threading
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt

from wdm_web_server.models.svm_lr_nb import svm
from wdm_web_server.models.svm_lr_nb import naivebayes
from wdm_web_server.models.svm_lr_nb import lr

from wdm_web_server.models.DecisionTree import DecisionTree
from wdm_web_server.models.perceptron_naive_bayes import perceptron
from wdm_web_server.models.Adaboost import adaboost


def get_result(lock, type, msg, result):
    if type == 1:
        algorithm, f1, classNum, confidence = svm.get_result(msg)
    elif type == 2:
        algorithm, f1, classNum, confidence = lr.get_result(msg)
    elif type == 3:
        algorithm, f1, classNum, confidence = naivebayes.get_result(msg)
    elif type == 4:
        algorithm, f1, classNum, confidence = DecisionTree.get_result(msg)
    elif type == 5:
        algorithm, f1, classNum, confidence = perceptron.get_result(msg)
    else:
        algorithm, f1, classNum, confidence = adaboost.get_result(msg)
    map = {}
    map['algorithm'] = algorithm
    map['f1'] = f1
    map['classNum'] = int(classNum)
    map['confidence'] = float(confidence)
    lock.acquire()
    result.append(map)
    lock.release()


@csrf_exempt
def classifier(request):
    lock = threading.Lock()

    result = []
    msg = request.POST.get('msg', '')
    if len(msg) == 0:
        msg = request.GET.get('msg', '')

    thread_list = []
    for i in range(1, 7):
        t = threading.Thread(target=get_result(lock, i, msg, result))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()

    # algorithm, f1, classNum, confidence = svm.get_result(msg)
    # map = {}
    # map['algorithm'] = algorithm
    # map['f1'] = f1
    # map['classNum'] = classNum
    # map['confidence'] = float(confidence)
    # result.append(map)
    #
    # algorithm, f1, classNum, confidence = lr.get_result(msg)
    # map = {}
    # map['algorithm'] = algorithm
    # map['f1'] = f1
    # map['classNum'] = classNum
    # map['confidence'] = float(confidence)
    # result.append(map)
    #
    # algorithm, f1, classNum, confidence = naivebayes.get_result(msg)
    # map = {}
    # map['algorithm'] = algorithm
    # map['f1'] = f1
    # map['classNum'] = classNum
    # map['confidence'] = float(confidence)
    # result.append(map)
    #
    # algorithm, f1, classNum, confidence = DecisionTree.get_result(msg)
    # map = {}
    # map['algorithm'] = algorithm
    # map['f1'] = f1
    # map['classNum'] = int(classNum)
    # map['confidence'] = float(confidence)
    # result.append(map)
    #
    # algorithm, f1, classNum, confidence = perceptron.get_result(msg)
    # map = {}
    # map['algorithm'] = algorithm
    # map['f1'] = f1
    # map['classNum'] = int(classNum)
    # map['confidence'] = float(confidence)
    # result.append(map)
    #
    # algorithm, f1, classNum, confidence = adaboost.get_result(msg)
    # map = {}
    # map['algorithm'] = algorithm
    # map['f1'] = f1
    # map['classNum'] = int(classNum)
    # map['confidence'] = float(confidence)
    # result.append(map)

    print(json.dumps(result))
    return HttpResponse(json.dumps(result), content_type="application/json")
