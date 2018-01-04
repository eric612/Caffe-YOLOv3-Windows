#!/usr/bin/env python

"""
Parse training log

Evolved from parse_log.sh
"""

import os
import re
import extract_seconds
import argparse
import csv
from collections import OrderedDict

import inspect
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """

    regex_iteration = re.compile('iter: (\d+)')
    regex_train_output = re.compile('loss: ([\.\deE+-]+)')
    regex_obj_score = re.compile('avg_obj: ([\.\deE+-]+)')
    regex_noobj = re.compile('avg_noobj: ([\.\deE+-]+)')
    regex_iou = re.compile('avg_iou: ([\.\deE+-]+)')
    regex_cat = re.compile('avg_cat: ([\.\deE+-]+)')
    regex_recall = re.compile('recall: ([\.\deE+-]+)')
    #regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    #regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')
    train_dict_list = []
    test_dict_list = []
    train_row = None
    test_row = None
    
    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)

        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            try:
                time = extract_seconds.extract_datetime_from_line(line,
                                                                  logfile_year)
            except ValueError:
                # Skip lines with bad formatting, for example when resuming solver
                continue

            seconds = (time - start_time).total_seconds()
	    
	    #learning_rate_match = regex_learning_rate.search(line)
            #if learning_rate_match:
            #    learning_rate = float(learning_rate_match.group(1))

            train_dict_list, train_row = parse_line_for_net_output(
                regex_train_output, regex_obj_score, regex_noobj, regex_iou, regex_cat, regex_recall, train_row, train_dict_list,
                line, iteration, seconds)
            #test_dict_list, test_row = parse_line_for_net_output(
            #    regex_test_output, test_row, test_dict_list,
            #    line, iteration, seconds, learning_rate
            #)
    	    
    #fix_initial_nan_learning_rate(train_dict_list)
    #fix_initial_nan_learning_rate(test_dict_list)

    return train_dict_list #, test_dict_list


def parse_line_for_net_output(regex_obj, regex_obj_score, regex_noobj, regex_iou, regex_cat, regex_recall, row, row_dict_list,
                              line, iteration, seconds):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """
    
    output_match = regex_obj.search(line)
    output_obj = regex_obj_score.search(line)
    output_noobj = regex_noobj.search(line)
    output_iou = regex_iou.search(line)
    output_cat = regex_cat.search(line)
    output_recall = regex_recall.search(line)    

    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds)
            ])
       
        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = 'loss'
        output_val = output_match.group(1) #the number of () in re.compile('loss: ([\.\deE+-]+)')
        row[output_name] = float(output_val)
    if output_obj:
    	avg_obj = output_obj.group(1)
	row['avg_obj'] = float(avg_obj)
    if output_noobj:
	row['avg_noobj'] = float(output_noobj.group(1)) 
    if output_iou:
	row['avg_iou'] = float(output_iou.group(1))
    if output_cat:
      	row['avg_cat'] = float(output_cat.group(1))
    if output_recall:
	row['recall'] = float(output_recall.group(1))    
		
    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']


def save_csv_files(logfile_path, output_dir, train_dict_list, #test_dict_list,
                   delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    print train_dict_list
    write_csv(train_filename, train_dict_list, delimiter, verbose)

    #test_filename = os.path.join(output_dir, log_basename + '.test')
    #write_csv(test_filename, test_dict_list, delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    step = 64;
    with open(data_file, 'r') as f:
	index = 0	
	p0 = list()
	p1 = list()	
	for line in f:
	    line = line.strip()
	    if line[0] != 'N':
                fields = line.split(',')
		#data[0].append(float(fields[field_idx0].strip()))
		#data[1].append(float(fields[field_idx1].strip()))
		p0.append(float(fields[field_idx0].strip()))
            	p1.append(float(fields[field_idx1].strip()))
	    index += 1
	    if index % step == 0:
		#print index
		data[0].append(p0[0] / step)
		data[1].append(sum(p1) / step)
		p0 = list()
		p1 = list() 
    #print len(data[0])
    return data

def plot_chart(data_file):
    field_index = {0: 'iters', 1: 'time', 2: 'loss', 3: 'avg_obj', 4: 'avg_noobj', 5: 'avg_iou', 6: 'avg_cat', 7: 'recall'}
    for i in range(2, 8):
        data = load_data(data_file, 0, i)
    	color = [random.random(), random.random(), random.random()]
    	plt.plot(data[0], data[1], label = '02', color = color,
                     linewidth = .75)
	plt.title(field_index[i]+'. vs ' + field_index[0] + '/64.')
	plt.xlabel(field_index[0]+'/64')
    	plt.ylabel(field_index[i])
    	plt.savefig('./fig/' + field_index[i]+'. vs ' + field_index[0] + '.png')
	plt.show()

def main():
    args = parse_args()
    if not os.path.exists(args.logfile_path + '.train'): 
    	train_dict_list = parse_log(args.logfile_path)
    	save_csv_files(args.logfile_path, args.output_dir, train_dict_list,
                   delimiter=args.delimiter)
    plot_chart(args.logfile_path+'.train')

if __name__ == '__main__':
    main()
