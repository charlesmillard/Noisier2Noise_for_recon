import numpy as np


def gen_pdf(nx, ny, delta, p, c_sq, sample_type):
    if sample_type == "bern":
        prob_map = gen_pdf_bern(nx, ny, delta, p, c_sq, 0)
    elif sample_type == "bern_inv":
        prob_map = gen_pdf_bern(nx, ny, delta, p, c_sq, 1)
    elif sample_type == "columns":
        prob_map = gen_pdf_columns(nx, ny, delta, p, c_sq)
    return prob_map


def gen_pdf_bern(nx, ny, delta, p, c_sq, inv_flag):
    # generates 2D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), sparse=False, indexing='xy')
    r = np.sqrt(xv ** 2 + yv ** 2)
    r /= np.max(r)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

    a = -1
    b = 1
    eta = 1e-3
    ii = 1
    while 1:
        c = a / 2 + b / 2
        prob_map = (1 - r) ** p + c
        prob_map[prob_map > 1] = 1
        prob_map[prob_map < 0] = 0

        if inv_flag:
            prob_map = 1 - prob_map

        prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

        delta_current = np.mean(prob_map)
        if delta > delta_current + eta:
            if inv_flag:
                b = c
            else:
                a = c
        elif delta < delta_current - eta:
            if inv_flag:
                a = c
            else:
                b = c
        else:
            break

        ii += 1
        if ii == 100:
            print('Careful - genPDF did not converge after 100 iterations')
            break

    return prob_map


def gen_pdf_columns(nx, ny, delta, p, c_sq):
    # generates 1D polynomial variable density with sampling factor delta, fully sampled central square c_sq
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
        c = (a + b) / 2
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


def mask_from_prob(prob_map, sample_type):
    prob_map[prob_map > 0.99] = 1
    if sample_type in ["bern", "bern_inv"]:
        mask = np.random.binomial(1, prob_map)
    elif sample_type == "columns":
        (nx, ny) = np.shape(prob_map)
        mask1d = np.random.binomial(1, prob_map[0:1])
        mask = np.repeat(mask1d, nx, axis=0)
    return np.array(mask, dtype=bool)