#-*- coding:utf-8 -*-

import click
import numpy as np
from data_loader import DataLoader
from model import Perceptron

def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))
    return 1 - mdiff

@click.command()
@click.option('--epoch_count', default=10, help='epoch')
@click.option('--mb_size', default=10, help='mini batch size')
@click.option('--report_seq', default=1, help='mini batch size')
def main(epoch_count, mb_size, report_seq):
    data_loader = DataLoader('./data/abalone.csv')

    data_loader.print_head1()

    model = Perceptron()

    step_count = data_loader.arrange_data(mb_size)

    print(f'epoch_count : {epoch_count}')
    print(f'mb_size : {mb_size}')
    print(f'step_count : {step_count}')

    test_x, test_y = data_loader.get_test_data()

    print(np.shape(test_y))

    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count):
            train_x, train_y = data_loader.get_train_data(mb_size, n)
            output, aux_nn = model.forward_neuralnet(train_x)

            # print(f'pred : {output[0]}')
            # print(f'label : {train_y[0]}')

            loss, aux_pp = model.forward_postproc(output, train_y)
            acc = eval_accuracy(output, train_y)

            G_loss = 1.0
            G_output = model.backprop_postproc(G_loss, aux_pp)
            model.backprop_neuralnet(G_output, aux_nn)

            losses.append(loss)
            accs.append(acc)

        if report_seq > 0 and (epoch + 1) % report_seq == 0:
            test_output, _ = model.forward_neuralnet(test_x)
            accuracy = eval_accuracy(test_output, test_y)
            print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'. \
                  format(epoch + 1, np.mean(losses), np.mean(accs), acc))

    ftest_output, _ = model.forward_neuralnet(test_x)
    final_acc = eval_accuracy(ftest_output, test_y)

    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

if __name__ == '__main__':
    main()

