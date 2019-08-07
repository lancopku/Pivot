import os
import pyrouge
import logging


def calc_rouge_score(candidate, reference, log_dir):
    assert len(reference) == len(candidate)

    ref_dir = os.path.join(log_dir, 'reference')
    cand_dir = os.path.join(log_dir, 'candidate')
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    for i in range(len(reference)):
        with open(os.path.join(ref_dir,"%06d_reference.txt" % i), 'w', encoding='utf-8') as f:
            f.write(reference[i] + '\n')
        with open(os.path.join(cand_dir,"%06d_candidate.txt" % i), 'w', encoding='utf-8') as f:
            f.write(candidate[i] + '\n')

    #rouge_args = '-c 95 -U -r 1 -n 2 -a'

    r = pyrouge.Rouge155()#rouge_args=rouge_args)
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    logging.getLogger('global').setLevel(logging.WARNING)
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_4_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_4_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_4_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    result = "F_measure: {0} Recall: {1} Precision: {2}\n".format(str(f_score), str(recall), str(precision))
    #print(result)
    
    output_dicts = {'score': {'rouge': f_score[1]}, 'logging': result}

    return output_dicts