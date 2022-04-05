import numpy as np

def genPDF(nx, ny, delta, p, c_sq, sample_type):

    if sample_type == "bern":
        prob_map = genPDFbern(nx, ny, delta, p, c_sq)
    elif sample_type == "columns":
        prob_map = genPDFcolumns(nx, ny, delta, p, c_sq)
    return prob_map

def genPDFbern(nx, ny, delta, p, c_sq):
    # generate polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), sparse=False, indexing='xy')

    r = np.sqrt(xv ** 2 + yv ** 2)
    r /= np.max(r)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

    a = 0
    b = 1

    eta = 1e-3

    ii = 1
    while 1:
        c = a / 2 + b / 2
        prob_map = (1 - r) ** p + c
        prob_map[prob_map > 1] = 1
        prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

        delta_current = np.mean(prob_map)
        if delta > delta_current + eta:
            a = c
        elif delta < delta_current - eta:
            b = c
        else:
            break

        ii += 1
        if ii == 100:
            print('Careful - genPDF did not converge after 100 iterations')
            break

    return prob_map

def genPDFcolumns(nx, ny, delta, p, c_sq):
    # generate polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, 1), np.linspace(-1, 1, ny), sparse=False, indexing='xy')

    r = np.abs(yv)
    r /= np.max(r)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

    a = -1
    b = 1

    eta = 1e-3

    ii = 1
    while 1:
        c = (a + b)/ 2
        prob_map = (1 - r) ** p + c
        prob_map[prob_map > 1] = 1
        prob_map[prob_map < 0] = 0
        prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

        delta_current = np.mean(prob_map)
        if delta > delta_current + eta:
            a = c
        elif delta < delta_current - eta:
            b = c
        else:
            break

        ii += 1
        if ii == 100:
            print('Careful - genRowsPDF did not converge after 100 iterations')
            break

    prob_map = np.repeat(prob_map, nx, axis=1)
    prob_map = np.rot90(prob_map)
    return prob_map

def maskFromProb(prob_map, sample_type):
    prob_map[prob_map > 0.99] = 1
    if sample_type == "bern":
        mask = np.random.binomial(1, prob_map)
    elif sample_type == "columns":
        (nx, ny) = np.shape(prob_map)
        mask1d = np.random.binomial(1, prob_map[0:1])
        mask = np.repeat(mask1d, nx, axis=0)

    return np.array(mask, dtype=bool)

def autoGenProbLambda(prob_omega, max_K, p, c, sample_type):
    nx, ny = prob_omega.shape
    trial_us_facs = np.linspace(0.01, 0.99, 100)
    largest_k = np.array([])
    for ii in range(len(trial_us_facs)):
        p_trial = genPDF(nx, ny, trial_us_facs[ii], p, c, sample_type)
        p_trial[p_trial > 0.99] = 0.99
        largest_k = np.append(largest_k, np.max((1 - prob_omega) / (1 - prob_omega * p_trial)))

    best_idx = np.sum(largest_k < max_K)
    if best_idx == 0:
        raise ValueError('Your maximum allowed K is too ambitious!')

    prob_lambda = genPDF(nx, ny, trial_us_facs[best_idx - 1], p, c, sample_type)
    prob_lambda[prob_lambda > 0.99] = 0.99

    print('automatically generated undersampling factor is {}'.format(np.mean(prob_lambda)))
    return prob_lambda