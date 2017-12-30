from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt

from wdm_web_server.models.svm_lr_nb import svm
from wdm_web_server.models.svm_lr_nb import naivebayes
from wdm_web_server.models.svm_lr_nb import lr


@csrf_exempt
def classifier(request):
    result = []
    msg = request.POST.get('msg', '')
    if len(msg) == 0:
        msg = request.GET.get('msg', '')

    algorithm, f1, classNum, confidence = svm.get_result(msg)
    map = {}
    map['algorithm'] = algorithm
    map['f1'] = f1
    map['classNum'] = classNum
    map['confidence'] = confidence
    result.append(map)

    algorithm, f1, classNum, confidence = lr.get_result(msg)
    map = {}
    map['algorithm'] = algorithm
    map['f1'] = f1
    map['classNum'] = classNum
    map['confidence'] = confidence
    result.append(map)

    algorithm, f1, classNum, confidence = naivebayes.get_result(msg)
    map = {}
    map['algorithm'] = algorithm
    map['f1'] = f1
    map['classNum'] = classNum
    map['confidence'] = confidence
    result.append(map)

    print(json.dumps(result))
    return HttpResponse(json.dumps(result), content_type="application/json")
