try:
    from .cec2017_py.cec2017.functions import all_functions
except ImportError:
    from cec2017_py.cec2017.functions import all_functions

class Feature:
    def __init__(self):
        self.all_function = all_functions.copy()
        self.descriptions = {}

    def getAllFuncName(self):
        for func in self.all_function:
            func_name = func.__name__
            doc = func.__doc__

            if doc:
                # 清理文档字符串：去除首尾空白、按换行分割
                doc_lines = [line.strip() for line in doc.strip().split('\n') if line.strip()]
                # 过滤掉包含"Args:"、"rotation ("、"shift ("的参数说明行
                describe = doc_lines[0]
            else:
                describe = "None"

            self.descriptions[func_name] = describe # {'f1': 'Shifted and Rotated Bent Cigar Function'}
