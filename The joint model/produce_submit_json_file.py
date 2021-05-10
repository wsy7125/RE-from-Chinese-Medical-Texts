from bin.integrated_model_output import Standard_Model_Output, get_predicate_matrix,\
    schemas_subject_object_type_2_predicate_list, predicate_id2label_map, get_predicate_labels

import json
import os
from tqdm import tqdm

#生成实体关系json文件的策略二
def generation_subject_predicate_object_triple_strategy_two(standard_model_output):
    # result_json_write_f = open("subject_predicate_object_predict_output.json", "w", encoding='utf-8')
    for text, entity_tuple_list, predicate_head in standard_model_output.text_entity_predicate_generator():
        predicate_head_id_matrix = get_predicate_matrix(predicate_head)
        print(text)#《不是所有时光都微笑》是2012年7月1日光明日报出版社出版的书籍，作者是蓝瞳
        print(entity_tuple_list)#[('图书作品', 2, '不是所有时光都微笑'), ('出版社', 19, '光明日报出版社'), ('人物', 35, '蓝瞳')]
        print(predicate_head_id_matrix.shape)#(128, 128, 50)
        print("\n")

        #Writing strategy function
        #  pass
        # return spo_list = "spo_list": [{"predicate": "毕业院校", "object_type": "学校", "subject_type": "人物", "object": "四川外语学院", "subject": "王润泽"}]

        # line_dict = dict()
        # line_dict["text"] = text
        # line_dict["spo_list"] = None
        # line_json = json.dumps(line_dict, ensure_ascii=False)
        # result_json_write_f.write(line_json + "\n")
    # result_json_write_f.close()


#生成实体关系json文件的策略一
def generation_subject_predicate_object_triple_strategy_one(standard_model_output, store_submit_json_file_path):
    # result_json_write_f = open(store_submit_json_file_path, "w", encoding='utf-8')
    #根据实体类型对应的可能关系类型以及模型输出的关系矩阵值生成候选关系列表
    def get_candicate_predicate_list(correct_predicate_scope_list, subject_object_predicate_row):
        candidate_predicate_list = list()
        predicate_label_list = get_predicate_labels()
        for predicate_label, predicate_row_value in zip(predicate_label_list, subject_object_predicate_row):
            if predicate_row_value > 0.5 and predicate_label in correct_predicate_scope_list:
                candidate_predicate_list.append(predicate_label)
        return candidate_predicate_list
    #只有这些类型可以做主体
    subject_type_list = [
        '疾病', '其他', '其他治疗', '手术治疗',
        '检查', '流行病学', '症状', '社会学',
        '药物', '部位']
    #只有这些类型可以做客体
    object_type_list = [
         '其他', '其他治疗', '手术治疗',
         '检查', '流行病学', '疾病', '症状',
         '社会学', '药物', '部位', '预后']
    def get_candicate_text_and_spo_list(model_output_tuple):
        text, entity_tuple_list, predicate_head = model_output_tuple[0], model_output_tuple[1], model_output_tuple[2]
        candicate_spo_list = list()
        predicate_head_id_matrix = get_predicate_matrix(predicate_head)
        for subject_entity_type, subject_entity_position, subject_entity_value in entity_tuple_list:
            for object_entity_type, object_entity_position, object_entity_value in entity_tuple_list:
                if subject_entity_type in subject_type_list and object_entity_type in object_type_list and subject_entity_value!= object_entity_value:
                    if (subject_entity_type, object_entity_type) in schemas_subject_object_type_2_predicate_list:
                        correct_predicate_scope_list = schemas_subject_object_type_2_predicate_list[
                            (subject_entity_type, object_entity_type)]
                        subject_object_predicate_row = predicate_head_id_matrix[subject_entity_position, object_entity_position]
                        candicate_predicate_list = get_candicate_predicate_list(correct_predicate_scope_list, subject_object_predicate_row)
                        for candicate_predicate in candicate_predicate_list:
                            print("1")
                            print(subject_entity_value,candicate_predicate,object_entity_value,subject_entity_type,object_entity_type)
                            candicate_spo_list.append({"subject": subject_entity_value, "predicate": candicate_predicate,
                                                       "object": object_entity_value, "subject_type": subject_entity_type,
                                                       "object_type": object_entity_type})
        return (text, candicate_spo_list)

    def get_candicate_text_and_spo_list_v2(text, entity_tuple_list, predicate_head):
        candicate_spo_list = list()
        predicate_head_id_matrix = get_predicate_matrix(predicate_head)
        for subject_entity_type, subject_entity_position, subject_entity_value in entity_tuple_list:
            for object_entity_type, object_entity_position, object_entity_value in entity_tuple_list:
                if subject_entity_type in subject_type_list and object_entity_type in object_type_list and subject_entity_value!= object_entity_value:
                    if (subject_entity_type, object_entity_type) in schemas_subject_object_type_2_predicate_list:
                        correct_predicate_scope_list = schemas_subject_object_type_2_predicate_list[
                            (subject_entity_type, object_entity_type)]
                        subject_object_predicate_row = predicate_head_id_matrix[subject_entity_position, object_entity_position]
                        candicate_predicate_list = get_candicate_predicate_list(correct_predicate_scope_list, subject_object_predicate_row)
                        for candicate_predicate in candicate_predicate_list:
                            print("1")
                            print(subject_entity_value,candicate_predicate,object_entity_value,subject_entity_type,object_entity_type)
                            candicate_spo_list.append({"subject": subject_entity_value, "predicate": candicate_predicate,
                                                       "object": object_entity_value, "subject_type": subject_entity_type,
                                                       "object_type": object_entity_type})
        return (text, candicate_spo_list)


    # text_and_spo_list = map(get_candicate_text_and_spo_list, standard_model_output.text_entity_predicate_generator()) #TODO:改为多进程提升速度
    text, entity_tuple_list, predicate_head = standard_model_output.text_entity_predicate()
    print(type(text))
    print(text)
    text_and_spo_list = get_candicate_text_and_spo_list_v2(text, entity_tuple_list, predicate_head)

    # for text, candicate_spo_list in tqdm(text_and_spo_list):
    #     spo_list = candicate_spo_list #TODO:增加校正函数
    #     line_dict = dict()
    #     line_dict["text"] = text
    #     line_dict["spo_list"] = spo_list
    #     line_json = json.dumps(line_dict, ensure_ascii=False)
    #     result_json_write_f.write(line_json + "\n")
    # result_json_write_f.close()

def get_store_submit_json_file_path(model_infer_out_file_path, store_file_dir, json_file_name="submit_entity_relation_file.json"):
    if not os.path.exists(store_file_dir):
        os.mkdir(store_file_dir)
    _, multiple_relations_model, epochs, ckpt = model_infer_out_file_path.split("/")
    json_file_name = epochs + "_" + ckpt + "_" + json_file_name
    json_file_name_path = os.path.join(store_file_dir, json_file_name)
    return json_file_name_path

if __name__=="__main__":
    model_infer_out_file_path = "infer_out/multiple_relations_model/epochs3/ckpt2000"
    standard_format_data_test_path = "bin/standard_format_data/test"
    store_submit_json_file_path = get_store_submit_json_file_path(model_infer_out_file_path, "store_submit_json_file")

    standard_model_output = Standard_Model_Output(model_infer_out_file_path, standard_format_data_test_path)
    # generation_subject_predicate_object_triple_strategy_one(standard_model_output, store_submit_json_file_path)
    generation_subject_predicate_object_triple_strategy_one(standard_model_output, store_submit_json_file_path)