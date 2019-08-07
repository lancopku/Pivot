import os
import logging
from tempfile import mkdtemp
import shutil
import subprocess
import re
import sys
import codecs

def calc_nist_score(candidate, reference, log_dir):
    assert len(reference) == len(candidate)

    data_ref = reference
    data_sys = candidate
    data_src = [''] * len(data_sys)

    temp_path = mkdtemp(prefix='e2e-eval-')

    # create MTEval files
    mteval_ref_file = os.path.join(temp_path, 'mteval_ref.sgm')
    create_mteval_file(data_ref, mteval_ref_file, 'ref')
    mteval_sys_file = os.path.join(temp_path, 'mteval_sys.sgm')
    create_mteval_file(data_sys, mteval_sys_file, 'tst')
    mteval_src_file = os.path.join(temp_path, 'mteval_src.sgm')
    create_mteval_file(data_src, mteval_src_file, 'src')
    mteval_log_file = os.path.join(temp_path, 'mteval_log.txt')

    mteval_path = 'metrics/mteval-v13a-sig.pl'
    mteval_out = subprocess.check_output(['perl', mteval_path,
                                          '-r', mteval_ref_file,
                                          '-s', mteval_src_file,
                                          '-t', mteval_sys_file,
                                          '-f', mteval_log_file], stderr=subprocess.STDOUT, encoding='utf-8')
    #print(mteval_out)
    
    #mteval_out = mteval_out.decode('utf-8')
    nist = float(re.search(r'NIST score = ([0-9.]+)', mteval_out).group(1))
    bleu = float(re.search(r'BLEU score = ([0-9.]+)', mteval_out).group(1))

    shutil.rmtree(temp_path)

    score = {'NIST': nist, 'BLEU': bleu}
    result = 'NIST: {0}, BLEU: {1}\n'.format(nist, bleu)
    
    output_dicts = {'score': score, 'logging': result}

    return output_dicts


def create_mteval_file(refs, path, file_type):
    """Given references/outputs, create a MTEval .sgm XML file.
    @param refs: data to store in the file (human references/system outputs/dummy sources)
    @param path: target path where the file will be stored
    @param file_type: the indicated "set type" (ref/tst/src)
    """
    # swap axes of multi-ref data (to 1st: different refs, 2nd: instances) & pad empty references
    data = [[]]
    for inst_no, inst in enumerate(refs):
        if not isinstance(inst, list):  # single-ref data
            inst = [inst]
        for ref_no, ref in enumerate(inst):
            if len(data) <= ref_no:  # there's more refs than previously known: pad with empty
                data.append([''] * inst_no)
            data[ref_no].append(ref)
        ref_no += 1
        while ref_no < len(data):  # less references than previously: pad with empty
            data[ref_no].append('')
            ref_no += 1

    with codecs.open(path, 'wb', 'UTF-8') as fh:
        settype = file_type + 'set'
        fh.write('<%s setid="%s" srclang="any" trglang="%s">\n' % (settype, 'e2e', 'en'))
        for inst_set_no, inst_set in enumerate(data):
            sysid = file_type + ('' if len(data) == 1 else '_%d' % inst_set_no)
            fh.write('<doc docid="test" genre="news" origlang="any" sysid="%s">\n<p>\n' % sysid)
            for inst_no, inst in enumerate(inst_set, start=1):
                fh.write('<seg id="%d">%s</seg>\n' % (inst_no, inst))
            fh.write('</p>\n</doc>\n')
        fh.write('</%s>' % settype)