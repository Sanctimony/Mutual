import os, logging, argparse
from copy import deepcopy
from tqdm import tqdm
import ujson as json
import spacy
nlp = spacy.load("en_core_web_sm")
import random



# create logger
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

answer_type_map = {'intra_entity_superlative': 'number', 'inter_entity_superlative': 'spans', 
                   'intra_entity_simple_diff': 'number', 'intra_entity_subset': 'number', 
                   'inter_entity_sum': 'number', 'inter_entity_comparison': 'spans', 'select': 'number'}

def convert_synthetic_texual_to_drop(data):
    _data = deepcopy(data)
    for sample in _data.values():
        for qa_pair in sample['qa_pairs']:
            qa_pair['query_id'] = qa_pair['question_id']
            a = {'number': '', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}
            if answer_type_map[qa_pair['generator']] == 'number':
                a['number'] = str(qa_pair['answer'])
            else:
                a['spans'].append(str(qa_pair['answer']))
            qa_pair['answer'] = a
            for key in list(qa_pair.keys()):
                if key not in {'query_id', 'question', 'answer'}:
                    del qa_pair[key]
    return _data

def convert_passage_to_multi_turn_article(passage):
    doc = nlp(passage)
    speaker = True
    if random.random()>0.5:
        speaker = False
    speaker_map = {True: 'm', False: 'f'}
    article = []
    for sent in doc.sents:
        article.append(speaker_map[speaker]+" : "+sent.text)
        article.append(" ")
        speaker ^= True
    return "".join(article[:-1]), speaker

def main():
    parser = argparse.ArgumentParser(description='For converting synthetic texual data to Mutual format.')
    parser.add_argument("--data_json", default='synthetic_textual_mixed_min3_max6_up0.7_train.json', type=str, 
                        help="The synthetic texual data .json file.")
    parser.add_argument("--save_path", default='../data/', type=str,
                        help="Save path for mutual format data.")
    args = parser.parse_args()

    logger.info("Reading %s" % args.data_json)
    with open(args.data_json, encoding='utf8') as f:
        data = json.load(f)

    logger.info("Converting...")

    final_data = []
    speaker_map = {True: 'm', False: 'f'}
    answer_map = {0:"A", 1:"B",}
    id = 7089
    for key, passage_with_qa in data.items():
        passage = passage_with_qa["passage"]
        qa_pairs = passage_with_qa["qa_pairs"]
        article, speaker  = convert_passage_to_multi_turn_article(passage)
        for qa_pair in qa_pairs:
            answer = qa_pair["answer"]
            # only use answers that are pure numbers for ease of generating other choices
            if not type(answer) == int:
                continue
            article_with_question = article+" "+ speaker_map[speaker]+ " : " + qa_pair["question"]
            answer = int(answer)
            answer_speaker = speaker_map[speaker ^ True]
            choices = [answer_speaker+" : "+str(answer)+".", answer_speaker+" : "+str(answer+1)+".", answer_speaker+" : "+str(answer-1)+".", answer_speaker+" : "+str(answer+5)+"."]
            order = [0,1,2,3]
            random.shuffle(order)
            shuffled_choices = []

            for i in order:
                shuffled_choices.append(choices[i])
            correct_answer = chr(ord("A")+order.index(0))
            output = {}
            output["answers"] = correct_answer
            output["options"] = shuffled_choices
            output["article"] = article_with_question
            output["id"] = "train_"+str(id)
            curr_save_path = args.save_path+"train_"+str(id)+".txt"
            with open(curr_save_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False)
            id+=1
        #     break
        # break
    print(id)

    
if __name__ == "__main__":
    main()

'''
python convert_synthetic_texual_to_mutual.py --data_json ../../data/synthetic_textual_mixed_min3_max6_up0.7_train.json
'''