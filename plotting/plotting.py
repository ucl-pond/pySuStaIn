# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
import numpy as np
from matplotlib import pyplot as plt
from mixture_model import ParametricMM

colors = ['C{}'.format(x) for x in range(10)]


def greedy_ascent_trace(greedy_dict):
    fig, ax = plt.subplots()
    for key, value in greedy_dict.items():
        scores = [x.score for x in value]
        iter_n = np.arange(len(scores))+1
        ax.plot(iter_n, scores, label='Init {}'.format(key+1))
    ax.legend(loc=0)
    fig.suptitle('Greedy Ascent Traces')
    return fig, ax


def mixture_model_grid(X, y, mixtures,
                       score_names=None, class_names=None, plotting_font_size=10):
                                              
    n_particp, n_biomarkers = X.shape
    if score_names is None:
        score_names = ['BM{}'.format(x+1) for x in range(n_biomarkers)]
    if class_names is None:
        class_names = ['Controls', 'Cases']
    n_x = np.round(np.sqrt(n_biomarkers)).astype(int)
    n_y = np.ceil(np.sqrt(n_biomarkers)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(10, 10))
    for i in range(n_biomarkers):
        bio_X = X[:, i]
        bio_y = y[~np.isnan(bio_X)]
        bio_X = bio_X[~np.isnan(bio_X)]

        hist_dat = [bio_X[bio_y == 0],
                    bio_X[bio_y == 1]]

        hist_c = colors[:2]
        leg1 = ax[i // n_x, i % n_x].hist(hist_dat,
                                          label=class_names,
                                          normed=True,
                                          color=hist_c,
                                          alpha=0.7,
                                          stacked=True)
        linspace = np.linspace(bio_X.min(), bio_X.max(), 100).reshape(-1, 1)
        if isinstance(mixtures[i], ParametricMM):
            controls_score, patholog_score = mixtures[i].pdf(mixtures[i].theta,
                                                             linspace)
        else:
            controls_score, patholog_score = mixtures[i].pdf(linspace)
        probability = 1-mixtures[i].probability(linspace)
        probability *= np.max((patholog_score, controls_score))
        ax[i // n_x, i % n_x].plot(linspace, controls_score, color=colors[0])
        ax[i // n_x, i % n_x].plot(linspace, patholog_score, color=colors[1])
        
        leg2 = ax[i // n_x, i % n_x].plot(linspace, probability, color=colors[4])
        
        ax[i // n_x, i % n_x].set_title(score_names[i], fontsize=plotting_font_size)
        ax[i // n_x, i % n_x].axes.get_yaxis().set_visible(False)
        
        plt.setp(ax[i // n_x, i % n_x].get_xticklabels(), fontsize=plotting_font_size)
 
    i += 1
    for j in range(i, n_x*n_y):
        fig.delaxes(ax[j // n_x, j % n_x])
               
    fig.legend(leg1[2]+leg2, list(class_names) + ['p(event occured)'],
               loc='lower right', fontsize=plotting_font_size)
    fig.tight_layout()
    return fig, ax


def mcmc_trace(mcmc_samples):
    scores = [x.score for x in mcmc_samples]
    iter_n = np.arange(len(scores))+1
    fig, ax = plt.subplots()
    ax.plot(iter_n, scores)
    ax.set_ylabel('Likelihood')
    ax.set_xlabel('Iteration Number')
    fig.suptitle('MCMC Trace')
    return fig, ax


def mcmc_uncert_mat(mcmc_samples, ml_order=None, score_names=None):
    if ml_order is None:
        ml_order = mcmc_samples[0].ordering
    else:
        ml_order = ml_order.ordering
    n_biomarkers = ml_order.shape[0]
    if score_names is None:
        score_names = ['BM{}'.format(x+1) for x in range(n_biomarkers)]
    all_orders = [x.ordering for x in mcmc_samples]
    all_orders = np.array(all_orders)
    confusion_mat = np.zeros((n_biomarkers, n_biomarkers))
    for i in range(n_biomarkers):
        confusion_mat[i, :] = np.sum(all_orders == ml_order[i], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(confusion_mat, interpolation='nearest', cmap='Purples')
    
    #* Added by Neil Oxtoby, June 2018
    if n_biomarkers > 8:
        stp = 2
    else:
        stp = 1
    tick_marks_x = np.arange(0,n_biomarkers,stp)
    labs = range(1, n_biomarkers+1,stp)
    ax.set_xticks(tick_marks_x)
    ax.set_xticklabels(labs, rotation=0)
    tick_marks_y = np.arange(n_biomarkers)
    ax.set_yticks(tick_marks_y+0.2)
    trimmed_scores = [x[2:].replace('_', ' ') if x.startswith('p_')
                      else x.replace('_', ' ') for x in score_names]
    ax.set_yticklabels(np.array(trimmed_scores, dtype='object')[ml_order],
                       rotation=30, ha='right',
                       rotation_mode='anchor')

    ax.set_ylabel('Biomarker Name', fontsize=20)
    ax.set_xlabel('Event Order', fontsize=20)
    
    # Minor ticks
    ax.set_xticks(np.arange(-.5, n_biomarkers, 1), minor=True);
    ax.set_yticks(np.arange(-.5, n_biomarkers, 1), minor=True);
    # Gridlines based on minor ticks
    ax.grid(False)
    #ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    #ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()

    return fig, ax


def stage_histogram(stages, y, max_stage=None, class_names=None):
    fig, ax = plt.subplots()
    hist_dat = [stages[y == 0],
                stages[y == 1]]
    if class_names is None:
        class_names = ['Controls', 'Patients']
    if max_stage is None:
        max_stage = stages.max()
    hist_c = colors[:2]
    n, bins, patch = ax.hist(hist_dat,
                             label=class_names,
                             normed=True,
                             color=hist_c,
                             stacked=False,
                             bins=max_stage+1)
    ax.legend(loc=0, fontsize=20)

    idxs = np.arange(max_stage+1)
    bin_w = bins[1] - bins[0]
    ax.set_xticks(bins+bin_w/2)
    ax.set_xticklabels([str(x) for x in idxs])

    ax.set_ylabel('Fraction', fontsize=20)
    ax.set_xlabel('EBM Stage', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=13)
    fig.tight_layout()
    return fig, ax
