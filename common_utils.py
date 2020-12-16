# Metrics function
from collections import OrderedDict
from aif360.metrics import ClassificationMetric

import matplotlib.pyplot as plt

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics

def plot_metrics(metrics_before, metrics_after, 
                 sensitive_attr='Race',
                 mitigation='Adversarial De-biasing', dataset='Compas'):
    total = 4; cols = 4
    rows = total //cols; rows += total % cols
    pos = range(1, total+1)
    
    # Plot styles
    plt.style.use('ggplot')
    options = {'font.size': 18,
           'figure.titlesize': 30}
    plt.rcParams.update(options)
    
    fig = plt.figure(figsize=(36,rows*8))
    x_pos = [1,2]
    kwargs = {
        'color': ['grey', 'teal'],
        'tick_label': ['Before mitigation', 'After mitigation']
    }
    
    axes_background = (0.9,0.9,0.9)
    i = 0; subplot_pad=20
    ax = fig.add_subplot(rows, cols, pos[i])
    ax.set_title('Classification Accuracy Difference', pad=subplot_pad)
    ax.bar(x_pos, [metrics_before.accuracy(), metrics_after.accuracy()], **kwargs)

    metrics = {'Demographic Parity Difference': 'mean_difference',
               'Disparate Impact': 'disparate_impact',
               'Equal Opportunity Difference': 'equal_opportunity_difference'}
    
    for title, methodname in  metrics.items():
        i += 1;
        ax = fig.add_subplot(rows, cols, pos[i])
        ax.yaxis.grid()
        ax.set_title(title, pad=subplot_pad)
        ax.set_ylim([-0.5,1])
        ax.axhline(0, color='k')
        ax.bar(x_pos, 
               [getattr(metrics_before, methodname)(), getattr(metrics_after, methodname)()],
               **kwargs)

    fig.subplots_adjust(bottom=0.2)
    title = 'Impact of {} on {} dataset - {}'.format(mitigation, dataset, sensitive_attr)
    fig.suptitle(title, y=0.08)
    fig.savefig('figures/'+title+'.png')
    plt.show()