import argparse
from mir_eval import transcription, io, util
import json
import numpy as np
import time

def prepare_data(answer_true, answer_pred, time_shift):
    ref_pitches = []
    ref_intervals = []
    est_intervals = []

    if time_shift >= 0.0:
        for i in range(len(answer_true)):
            if (answer_true[i] is not None and float(answer_true[i][1]) - float(answer_true[i][0]) > 0
                and answer_true[i][0] >= 0.0):
                ref_intervals.append([answer_true[i][0], answer_true[i][1]])
                ref_pitches.append(answer_true[i][2])

        for i in range(len(answer_pred)):
            if (answer_pred[i] is not None and float(answer_pred[i][1]) - float(answer_pred[i][0]) > 0 
                and answer_pred[i][0]+time_shift >= 0.0):
                est_intervals.append([answer_pred[i][0]+time_shift, answer_pred[i][1]+time_shift])

    else:
        for i in range(len(answer_true)):
            if (answer_true[i] is not None and float(answer_true[i][1]) - float(answer_true[i][0]) > 0
                and answer_true[i][0]-time_shift >= 0.0):
                ref_intervals.append([answer_true[i][0]-time_shift, answer_true[i][1]-time_shift])
                ref_pitches.append(answer_true[i][2])

        for i in range(len(answer_pred)):
            if (answer_pred[i] is not None and float(answer_pred[i][1]) - float(answer_pred[i][0]) > 0
                and answer_pred[i][0] >= 0.0):
                est_intervals.append([answer_pred[i][0], answer_pred[i][1]])

    ref_intervals = np.array(ref_intervals)
    est_intervals = np.array(est_intervals)

    return ref_intervals, est_intervals, ref_pitches



def correct_one_data(answer_true, unvoiced, diff_list, correct_tolerance=0):
    
    ref_intervals, unvoiced_intervals, ref_pitches = prepare_data(answer_true, unvoiced, time_shift=0)

    in_range = 0 # within unvoiced frame
    out_range = 0 # -0.1~+0.1s
    total = len(ref_intervals)

    ref_intervals2 = []
    unvoiced_intervals2 = []
    j = 0
    for i in range(len(ref_intervals)):
        if j == len(unvoiced_intervals):
            break
        while unvoiced_intervals[j][1] + (correct_tolerance+0.0001) < ref_intervals[i][0] and j < len(unvoiced_intervals)-1:
            j = j + 1
        if unvoiced_intervals[j][0] - (correct_tolerance+0.0001) > ref_intervals[i][0]:
            continue
        if (unvoiced_intervals[j][1] + (correct_tolerance+0.0001) > ref_intervals[i][0] 
            and unvoiced_intervals[j][0] - (correct_tolerance+0.0001) < ref_intervals[i][0]):

            if (unvoiced_intervals[j][1] < ref_intervals[i][0] 
                and ref_intervals[i][0] - unvoiced_intervals[j][1] < correct_tolerance):
                # print (answer_true[i][0], unvoiced_intervals[j][1])
                answer_true[i][0] = unvoiced_intervals[j][1]
                # print (answer_true[i][0], unvoiced_intervals[j][1])
                diff_list.append(ref_intervals[i][0] - unvoiced_intervals[j][1])

            elif (unvoiced_intervals[j][0] > ref_intervals[i][0]
                and unvoiced_intervals[j][0] - ref_intervals[i][0] < correct_tolerance):
                answer_true[i][0] = unvoiced_intervals[j][0]
                diff_list.append(unvoiced_intervals[j][0] - ref_intervals[i][0])

            else:
                diff_list.append(0.0)

            in_range = in_range + 1
            ref_intervals2.append(i)
            unvoiced_intervals2.append(j)
            j = j + 1

    return answer_true

def post_process(unvoiced):
    new_unvoiced = []
    for i in range(len(unvoiced)):
        if i == 0 or unvoiced[i][0] - new_unvoiced[-1][1] > 0.09:
            new_unvoiced.append(unvoiced[i])
        else:
            new_unvoiced[-1][1] = unvoiced[i][1]

    return new_unvoiced

def correct_all(answer_true, unvoiced, output_path, correct_tolerance=0.05, shifting=0, print_result=True, id_list=None):

    diff_list = []
    total_seg = 0
    in_range_list = np.zeros(11)

    output = {}
    if True:
        result = np.zeros(3)
        for i in range(len(answer_true)):
            # unvoiced[i] = post_process(unvoiced[i])
            total_seg = total_seg + len(unvoiced[i])
            trial = [0.0,]
            output[id_list[i]] = correct_one_data(answer_true[i], unvoiced[i], diff_list, correct_tolerance=correct_tolerance)
            

    with open(output_path, 'w') as f:
        output_string = json.dumps(output)
        f.write(output_string)

    return result


class CorrectOnset():
    def __init__(self):
        self.orig_labels = None
        self.unvoiced = None

    def prepare_data(self, orig_label_path, unvoiced_path):
        with open(unvoiced_path) as json_data:
            unvoiced = json.load(json_data)

        with open(orig_label_path) as json_data:
            orig_labels = json.load(json_data)

        length = len(unvoiced)
        orig_labels_data = []
        unvoiced_data = []
        id_list = []
        for i in unvoiced.keys():
            if i in orig_labels.keys():
                orig_labels_data.append(orig_labels[i])
                unvoiced_data.append(unvoiced[i])
                id_list.append(i)

        self.orig_labels = orig_labels_data
        self.unvoiced = unvoiced_data
        self.id_list = id_list

    def correct_all(self, correct_tolerance, output_path):
        return correct_all(self.orig_labels, self.unvoiced, output_path=output_path, correct_tolerance=correct_tolerance, id_list=self.id_list)


def main(args):
    my_co = CorrectOnset()
    my_co.prepare_data(args.orig_label_file, args.unvoiced_file)
    print (time.time())
    my_co.correct_all(correct_tolerance=float(args.tol), output_path=args.output_path)
    print (time.time())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('orig_label_file')
    parser.add_argument('unvoiced_file')
    parser.add_argument('output_path')
    parser.add_argument('tol')

    args = parser.parse_args()

    main(args)
