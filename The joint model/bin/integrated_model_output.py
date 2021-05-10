import os
import numpy as np

schemas_subject_object_type_2_predicate_list = {
    ('疾病', '其他'): ['预防','阶段','就诊科室'],
    ('其他', '其他'): ['同义词'],
    ('疾病', '其他治疗'): ['辅助治疗','化疗','放射治疗'],
    ('其他治疗', '其他治疗'): ['同义词'],
    ('疾病', '手术治疗'): ['手术治疗'],
    ('手术治疗', '手术治疗'): ['同义词'],
    ('疾病', '检查'): ['实验室检查','影像学检查','辅助检查','组织学检查','内窥镜检查','筛查'],
    ('检查', '检查'): ['同义词'],
    ('疾病', '流行病学'): ['多发群体','发病率','发病年龄','多发地区','发病性别倾向','死亡率','多发季节','传播途径'],
    ('流行病学', '流行病学'): ['同义词'],
    ('疾病', '疾病'): ['同义词','并发症','病理分型','相关（导致）','鉴别诊断','相关（转化）','相关（症状）'],
    ('疾病', '症状'): ['临床表现','治疗后症状','侵及周围组织转移的症状'],
    ('症状', '症状'): ['同义词'],
    ('疾病', '社会学'): ['病因','高危因素','风险评估因素','病史','遗传因素','发病机制','病理生理'],
    ('疾病', '药物'): ['药物治疗'],
    ('药物', '药物'): ['同义词'],
    ('疾病', '部位'): ['发病部位','转移部位','外侵部位'],
    ('部位', '部位'): ['同义词'],
    ('疾病', '预后'): ['预后状况','预后生存率']}


def get_predicate_labels():
    "N --> no predicate"
    return ["N", '预防', '阶段', '就诊科室', '同义词', '辅助治疗', '化疗', '放射治疗', '同义词', '手术治疗', '同义词', '实验室检查', '影像学检查', '辅助检查', '组织学检查','同义词', '内窥镜检查', '筛查', '多发群体', '发病率', '发病年龄', '多发地区', '发病性别倾向', '死亡率', '多发季节', '传播途径', '同义词', '同义词', '并发症', '病理分型','相关（导致）', '鉴别诊断', '相关（转化）', '相关（症状）', '临床表现', '治疗后症状', '侵及周围组织转移的症状', '同义词', '病因', '高危因素', '风险评估因素', '病史', '遗传因素', '同义词', '发病机制', '病理生理','药物治疗', '同义词', '发病部位', 
    '转移部位', '外侵部位', '同义词', '预后状况', '预后生存率']

def predicate_id2label_map():
    predicate_label_list = ["N", '预防', '阶段', '就诊科室', '同义词', '辅助治疗', '化疗', '放射治疗', '同义词', '手术治疗', '同义词', '实验室检查', '影像学检查', '辅助检查', '组织学检查','同义词', '内窥镜检查', '筛查', '多发群体', '发病率', '发病年龄', '多发地区', '发病性别倾向', '死亡率', '多发季节', '传播途径', '同义词', '同义词', '并发症', '病理分型','相关（导致）', '鉴别诊断', '相关（转化）', '相关（症状）', '临床表现', '治疗后症状', '侵及周围组织转移的症状', '同义词', '病因', '高危因素', '风险评估因素', '病史', '遗传因素', '同义词', '发病机制', '病理生理','药物治疗', '同义词', '发病部位', 
    '转移部位', '外侵部位', '同义词', '预后状况', '预后生存率']
    predicate_label_id2label = dict()
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_id2label[i] = label
    return predicate_label_id2label


class Standard_Model_Output(object):
    """
    INPUT:
    model_infer_out_file_path
    standard_format_data_test_path
    OUTPUT:
    text 相传西汉的时候，汉哀帝很宠幸自己当朝臣子的儿子董贤，他下朝的时候注意到董贤，是因为董贤长得标致
    entity_tuple_list [('Text', 3, '西汉'), ('历史人物', 24, '董贤')]
    predicate_head_id_matrix.shape (128, 128, 50)
    """
    def __init__(self, model_infer_out_file_path, standard_format_data_test_path):
        """
        :param model_infer_out_file_path:
        :param standard_format_data_test_path:
        """
        self.model_infer_out_file_path = model_infer_out_file_path
        self.standard_format_data_test_path = standard_format_data_test_path

    #获取输入文件路径
    def get_input_file_path(self):
        token_label_predictions_file_path = os.path.join(self.model_infer_out_file_path, "token_label_predictions.txt")
        predicate_head_predictions_file_path = os.path.join(self.model_infer_out_file_path, "predicate_head_predictions_id.txt")
        predicate_head_probabilities_file_path = os.path.join(self.model_infer_out_file_path, "predicate_head_probabilities.txt")
        text_file_path = os.path.join(self.standard_format_data_test_path, "text.txt")
        token_in_not_UNK_file_path = os.path.join(self.standard_format_data_test_path, "token_in_not_UNK.txt")
        return token_label_predictions_file_path, predicate_head_predictions_file_path, \
               predicate_head_probabilities_file_path, text_file_path, token_in_not_UNK_file_path

    # 把模型输出实体标签位置和内容
    def model_token_label_2_entity_sort_tuple_list(self, token_in_not_UNK_list, predicate_token_label_list):
        """
        :param token_in_not_UNK:  ['紫', '菊', '花', '草', '是', '菊', '目', '，', '菊', '科', '，', '松', '果', '菊', '属', '的', '植', '物']
        :param predicate_token_label: ['B-生物', 'I-生物', 'I-生物', 'I-生物', 'O', 'B-目', 'I-目', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        :return: [('生物', '1', '紫菊花草'), ('目', '6', '菊目')]
        """
        #合并由WordPiece切分的词和单字
        def _merge_WordPiece_and_single_word(entity_sort_list):
            # [..['B-SUB', '新', '地', '球', 'ge', '##nes', '##is'] ..]---> [..('SUB', '新地球genesis')..]
            entity_sort_tuple_list = []
            for a_entity_list in entity_sort_list:
                entity_content = ""
                entity_type = None
                for idx, entity_part in enumerate(a_entity_list):
                    if idx == 0:
                        entity_type = entity_part
                        if entity_type[:2] not in ["B-", "I-"]:
                            break
                    else:
                        if entity_part.startswith("##"):
                            entity_content += entity_part.replace("##", "")
                        else:
                            entity_content += entity_part
                if entity_content != "":
                    entity_sort_tuple_list.append((entity_type[2:], entity_content))
            return entity_sort_tuple_list
        # 除去模型输出的特殊符号
        def preprocessing_model_token_lable(predicate_token_label_list, token_in_list_lenth):
            # ToDo:检查错误，纠错
            if predicate_token_label_list[0] == "[CLS]":
                predicate_token_label_list = predicate_token_label_list[1:]  # y_predict.remove('[CLS]')
            if len(predicate_token_label_list) > token_in_list_lenth:  # 只取输入序列长度即可
                predicate_token_label_list = predicate_token_label_list[:token_in_list_lenth]
            return predicate_token_label_list
        # 预处理标注数据列表
        predicate_token_label_list = preprocessing_model_token_lable(predicate_token_label_list, len(token_in_not_UNK_list))
        entity_sort_list = []
        entity_part_list = []
        #TODO:需要检查以下的逻辑判断，可能写的不够完备充分
        for idx, token_label in enumerate(predicate_token_label_list):
            # 如果标签为 "O"
            if token_label == "O":
                # entity_part_list 不为空，则直接提交
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
            # 如果标签以字符 "B-" 开始
            if token_label.startswith("B-"):
                # 如果 entity_part_list 不为空，则先提交原来 entity_part_list
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
                entity_part_list.append(token_label+ "_head_index_"+ str(idx + 1))
                entity_part_list.append(token_in_not_UNK_list[idx])
                # 如果到了标签序列最后一个标签处
                if idx == len(predicate_token_label_list) - 1:
                    entity_sort_list.append(entity_part_list)
            # 如果标签以字符 "I-"  开始 或者等于 "[##WordPiece]"
            if token_label.startswith("I-") or token_label == "[##WordPiece]":
                # entity_part_list 不为空，则把该标签对应的内容并入 entity_part_list
                if len(entity_part_list) > 0:
                    entity_part_list.append(token_in_not_UNK_list[idx])
                    # 如果到了标签序列最后一个标签处
                    if idx == len(predicate_token_label_list) - 1:
                        entity_sort_list.append(entity_part_list)
            # 如果遇到 [SEP] 分隔符，说明需要处理的标注部分已经结束
            if token_label == "[SEP]":
                break
        #[('影视作品_head_index_2', '鬼影实录2'), ('人物_head_index_9', '托德·威廉姆斯'), ('人物_head_index_19', '布赖恩·波兰德')]
        entity_sort_tuple_list = _merge_WordPiece_and_single_word(entity_sort_list)
        #拆分实体类型和所在原句中的序号
        def split_entity_type_and_predicate_index(entity_type_and_predicate_index_str):
            entity_type = entity_type_and_predicate_index_str[:entity_type_and_predicate_index_str.find("_head_index_")]
            predicate_index = entity_type_and_predicate_index_str[entity_type_and_predicate_index_str.find("_head_index_") + len("_head_index_"):]
            predicate_index = int(predicate_index)
            return entity_type, predicate_index
        #[(('影视作品', 2), '鬼影实录2'), (('人物', 9), '托德·威廉姆斯'), (('人物', 19), '布赖恩·波兰德')]
        entity_sort_tuple_list = [(split_entity_type_and_predicate_index(entity_type_and_predicate_index_str), entity) for entity_type_and_predicate_index_str, entity in entity_sort_tuple_list]
        #[('影视作品', 2, '鬼影实录2'), ('人物', 9, '托德·威廉姆斯'), ('人物', 19, '布赖恩·波兰德')]
        entity_sort_tuple_list = [(type_index[0], type_index[1], entity) for type_index, entity in entity_sort_tuple_list]
        return entity_sort_tuple_list

    #句子、实体元组列表、关系矩阵的生成器
    def text_entity_predicate_generator(self):
        token_label_predictions_file_path, predicate_head_predictions_file_path, \
        predicate_head_probabilities_file_path, text_file_path, token_in_not_UNK_file_path = self.get_input_file_path()
        token_label_predictions_file = open(token_label_predictions_file_path, "r", encoding='utf-8')
        predicate_head_predictions_file = open(predicate_head_predictions_file_path, "r", encoding='utf-8')
        text_file = open(text_file_path, "r", encoding='utf-8')
        token_in_not_UNK_file= open(token_in_not_UNK_file_path, "r", encoding='utf-8')
        for text, token_in_not_UNK, token_label, predicate_head in zip(text_file, token_in_not_UNK_file, token_label_predictions_file, predicate_head_predictions_file):
            text = text.replace("\n", "")
            token_in_not_UNK_list = token_in_not_UNK.replace("\n", "").split(" ")
            token_label_list = token_label.replace("\n", "").split(" ")
            entity_tuple_list = self.model_token_label_2_entity_sort_tuple_list(token_in_not_UNK_list, token_label_list)
            yield (text, entity_tuple_list, predicate_head)

    #句子、实体元组列表、关系矩阵的生成器
    def text_entity_predicate(self):
        # model_output_tuple = []
        token_label_predictions_file_path, predicate_head_predictions_file_path, \
        predicate_head_probabilities_file_path, text_file_path, token_in_not_UNK_file_path = self.get_input_file_path()
        token_label_predictions_file = open(token_label_predictions_file_path, "r", encoding='utf-8')
        predicate_head_predictions_file = open(predicate_head_predictions_file_path, "r", encoding='utf-8')
        text_file = open(text_file_path, "r", encoding='utf-8')
        token_in_not_UNK_file= open(token_in_not_UNK_file_path, "r", encoding='utf-8')
        for text, token_in_not_UNK, token_label, predicate_head in zip(text_file, token_in_not_UNK_file, token_label_predictions_file, predicate_head_predictions_file):
            text = text.replace("\n", "")
            token_in_not_UNK_list = token_in_not_UNK.replace("\n", "").split(" ")
            token_label_list = token_label.replace("\n", "").split(" ")
            entity_tuple_list = self.model_token_label_2_entity_sort_tuple_list(token_in_not_UNK_list, token_label_list)
            # model_output_tuple[0] = text
            # model_output_tuple[1] = entity_tuple_list
            # model_output_tuple[2] = predicate_head
            return text, entity_tuple_list, predicate_head

def get_predicate_matrix(predicate_head):
    predicate_head_id_list = predicate_head.replace("\n", "").split(" ")
    predicate_head_id_list = [float(id) for id in predicate_head_id_list]
    predicate_head_id_matrix = np.array(predicate_head_id_list).reshape((128, 128, 54))
    return predicate_head_id_matrix

if __name__=="__main__":
    model_infer_out_file_path = "../infer_out/multiple_relations_model/epochs3/ckpt2000"
    standard_format_data_test_path = "standard_format_data/test"
    standard_model_output = Standard_Model_Output(model_infer_out_file_path, standard_format_data_test_path)
    for text, entity_tuple_list, predicate_head in standard_model_output.text_entity_predicate_generator():
        predicate_head_id_matrix = get_predicate_matrix(predicate_head)
        print(text)
        print(entity_tuple_list)
        print(predicate_head_id_matrix.shape)
        print("\n")
