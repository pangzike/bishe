import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast
import random
from rank_bm25 import BM25Okapi
from utils.utils_general import *

relation_set = ["address", "area", "food", "phone", "postcode",
                "pricerange", "stars", "type", "choice", "ref", "name", "entrance_fee"]

# get 7 entity random from all entity
def generate_random_7_db(all_db):
    # 随机抽取7个实体
    random_7_db = random.sample(all_db, 7)
    return random_7_db

# take all attribute of entity
def get_all_db_attribute(all_db):
    all_db_attribute = [
        [v for k, v in entity.items()]
        for entity in all_db
    ]
    return all_db_attribute

# get 7 entity by BM25 from all entity
def generate_bm25_db(all_db, dialog_origin):
    '''
    all_db: all entity
    dialog: dialog content in one list
    '''

    dialog = []
    for i in dialog_origin:
        dialog.extend(i)

    # get all attribute of all entity
    all_db_attribute = get_all_db_attribute(all_db)
    corups = []
    for entity_attribute in all_db_attribute:
        attribute = [v.replace("_", " ") for v in entity_attribute]
        # 分词
        attribute = " ".join(attribute).split()
        corups.append(attribute)
    bm25 = BM25Okapi(corups)
    # 相似度从大到小排序，取相似度最大的7个
    scores = bm25.get_scores(" ".join(dialog).split())
    # 取相似度最大的7个
    scores_index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:7]

    bm25_db = [all_db[i] for i in scores_index]
    return bm25_db

# get 7 entity by frequency from all entity
# 实体的attribute作为corups，dialog作为query，
def generate_frequency_db(all_db, dialog_origin):
    # get all attribute of all entity
    # 把dialog拼成一个字符串
    dialog = []
    for i in dialog_origin:
        dialog.extend(i)

    dialog = " ".join(dialog)
    frequency_score = [0]*len(all_db)
    all_db_attribute = get_all_db_attribute(all_db)
    all_db_attribute = [[v.replace("_", " ") for v in entity_attribute] for entity_attribute in all_db_attribute]
    for i, entity in enumerate(all_db_attribute):
        for attribute in entity:
            if attribute in dialog:
                frequency_score[i] += 1
    # 取frequency_score最大的7个
    frequency_index = sorted(range(len(frequency_score)), key=lambda k: frequency_score[k], reverse=True)[:7]
    frequency_db = [all_db[i] for i in frequency_index]
    return frequency_db

def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, kb_arr, kb_id = [], [], [], []
    max_resp_len = 0
    choice, ref = None, None
    with open('data/MULTIWOZ2.1/global_entities.json') as f:
        global_entity = json.load(f)
        global_entity_list = {}
        for key in global_entity.keys():
            if key not in global_entity_list:
                global_entity_list[key] = []
            global_entity_list[key] += [item.lower().replace(' ', '_') for item in global_entity[key]]

    with open('data/MULTIWOZ2.1/data_db/all_db_clean_with_domain.json') as f:
        all_db = json.load(f)
    all_db_entity, all_db_entity_type = [], []
    # all_db_entity['attraction'] = []
    # all_db_entity['hotel'] = []
    # all_db_entity['restaurant'] = []
    # all_db_entity_type['attraction'] = []
    # all_db_entity_type['hotel'] = []
    # all_db_entity_type['restaurant'] = []

    # for entity in all_db:
    #     # task_type = entity['domain']
    #     for key in entity.keys():
    #         if key == 'internet' or key == 'parking' or key == 'domain':
    #             continue
    #         if entity[key] not in all_db_entity:
    #             all_db_entity.append(entity[key])
    #             all_db_entity_type.append('@'+key)

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if "#" == line[0]:
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    context_arr.append(u.split(' '))

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_restaurant, ent_idx_attraction, ent_idx_hotel = [], [], []
                    if task_type == "restaurant":
                        ent_idx_restaurant = gold_ent
                    elif task_type == "attraction":
                        ent_idx_attraction = gold_ent
                    elif task_type == "hotel":
                        ent_idx_hotel = gold_ent
                    ent_index = list(set(ent_idx_restaurant + ent_idx_attraction + ent_idx_hotel))

                    # Get entity set
                    entity_set, entity_type_set = [], []
                    # this_turn_entity = generate_random_7_db(all_db)
                    this_turn_entity = generate_bm25_db(all_db, context_arr)
                    # this_turn_entity = generate_frequency_db(all_db, context_arr)

                    for entity in this_turn_entity:
                        # task_type = entity['domain']
                        for key in entity.keys():
                            if key == 'internet' or key == 'parking' or key == 'domain':
                                continue
                            if entity[key] not in entity_set:
                                entity_set.append(entity[key])
                                entity_type_set.append('@'+key)

                    # entity_set, entity_type_set = generate_entity_set(kb_arr)
                    if choice is not None:
                        entity_set.append(choice)
                        entity_type_set.append('@choice')
                    if ref is not None:
                        entity_set.append(ref)
                        entity_type_set.append('@ref')

                    entity_set, entity_type_set = generate_entity_from_context(context_arr, global_entity_list, entity_set, entity_type_set)

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        if key in entity_set:
                            index = entity_set.index(key)
                        else:
                            index = len(entity_set)
                        ptr_index.append(index)

                    sketch_response = generate_template(global_entity_list, r, gold_ent, entity_set, entity_type_set,
                                                        task_type)
                    # add empty token
                    if len(entity_set) == 0:
                        entity_set.append("$$$$")
                        entity_type_set.append("empty_token")

                    entity_set.append("$$$$")
                    entity_type_set.append("empty_token")

                    #generate indicator
                    indicator = generate_indicator(context_arr, entity_set)

                    # generate graph
                    graph = generate_graph(entity_set, relation_set, kb_arr, this_turn_entity, choice, ref, task_type)

                    data_detail = {
                        'context_arr': list(context_arr),
                        'kb_arr': list(entity_set),
                        'response': r.split(' '),
                        'sketch_response': sketch_response.split(' '),
                        'ptr_index': ptr_index + [len(entity_set) - 1],
                        'indicator': indicator,
                        'ent_index': ent_index,
                        'ent_idx_res': list(set(ent_idx_restaurant)),
                        'ent_idx_att': list(set(ent_idx_attraction)),
                        'ent_idx_hot': list(set(ent_idx_hotel)),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type,
                        'graph': graph}
                    data.append(data_detail)

                    context_arr.append(r.split(' '))

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    kb_id.append(nid)
                    kb_info = line.split(' ')
                    if 'choice' in kb_info:
                        choice = kb_info[-1]
                    if 'ref' in kb_info:
                        ref = kb_info[-1]
                    kb_arr.append(kb_info)
            else:
                cnt_lin += 1

                if cnt_lin > 100:
                    break
                context_arr, kb_arr, kb_id = [], [], []
                choice, ref = None, None
                if(max_line and cnt_lin >= max_line):
                    break
    return data, max_resp_len


def generate_graph(entity_set, relation_set, kb_arr, all_db, choice, ref, domain):
    node_num = len(entity_set)
    edge_num = len(relation_set)

    # check validity
    # for kb in kb_arr:
    #     if 'name' in kb:
    #         continue
    #     assert kb[1] in relation_set, kb[1]

    graph = []
    for entity in all_db:
        # if domain != entity['domain']:
        #     continue

        for item in entity.keys():
            if item == 'name' or item == 'internet' or item == 'parking' or item == 'domain':
                continue
            entity_id1 = entity_set.index(entity['name'])
            relation_id = relation_set.index(item)
            entity_id2 = entity_set.index(entity[item])
            graph.append([relation_id, entity_id1, entity_id2])
            graph.append([relation_id, entity_id2, entity_id1])
        if choice is not None:
            entity_id1 = entity_set.index(entity['name'])
            relation_id = relation_set.index('choice')
            entity_id2 = entity_set.index(choice)
            graph.append([relation_id, entity_id1, entity_id2])
            graph.append([relation_id, entity_id2, entity_id1])
        if ref is not None:
            entity_id1 = entity_set.index(entity['name'])
            relation_id = relation_set.index('ref')
            entity_id2 = entity_set.index(ref)
            graph.append([relation_id, entity_id1, entity_id2])
            graph.append([relation_id, entity_id2, entity_id1])

    if len(graph) == 0:
        graph.append([0, 0, 0])

    return graph

def generate_indicator(context_arr, entity_set):
    """
    generate a list with the same size of context_arr, indicating whether each element of context_arr appears in kb_arr
    """

    indicator = []
    for s_id, question in enumerate(context_arr):
        indicator.append([1 if entity in question else 0 for entity in entity_set])

    return indicator

def generate_template(global_entity, sentence, sent_ent, entity_set, entity_type_set, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for entity_id, entity in enumerate(entity_set):
                    if word == entity:
                        ent_type = entity_type_set[entity_id]
                        break
                if ent_type == None:
                    for k, v in global_entity.items():
                        if word in v:
                            ent_type = k
                            break

                sketch_response.append('@'+ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response

def generate_entity_from_context(context_arr, global_entity, entity_set, entity_type_set):
    for sent in context_arr:
        for entity in sent:
            if entity in entity_set:
                continue
            for k, v in global_entity.items():
                if entity in v:
                    entity_set.append(entity)
                    entity_type_set.append(k)
                    break
    return entity_set, entity_type_set


def generate_entity_set(kb_arr):
    entity_set, entity_type_set = [], []
    for kb in kb_arr:
        if 'name' in kb:
            continue
        if kb[0] not in entity_set:
            entity_set.append(kb[0])
            entity_type_set.append('name')
        if kb[2] not in entity_set:
            entity_set.append(kb[2])
            entity_type_set.append(kb[1])

    return entity_set, entity_type_set


def prepare_data_seq(batch_size=100):
    file_train = 'data/MULTIWOZ2.1/train.txt'
    file_dev = 'data/MULTIWOZ2.1/dev.txt'
    file_test = 'data/MULTIWOZ2.1/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True, len(relation_set))
    dev = get_seq(pair_dev, lang, batch_size, False, len(relation_set))
    test = get_seq(pair_test, lang, batch_size, False, len(relation_set))

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, lang, max_resp_len, len(relation_set)


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d
