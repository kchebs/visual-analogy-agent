# Solves 2x2 and 3x3 Raven's Prog. Mat.
import numpy as num
from PIL import ImageChops as PIC, Image as PIM

def fill_amt(fgr):
    if hasattr(fgr, 'visualFilename'):
        img = PIM.open(fgr.visualFilename)
    else:
        img = fgr
    fill = 0
    wdh, hgt = img.size
    pxls = img.load()
    for h in range(hgt):
        for w in range(wdh):
            if (0, 0, 0, 255) == pxls[w, h]:
                fill += 1
    return fill

def rot180_d(fg1, fg2):
    image_one, image_two = opn_img(fg1, fg2)
    # Rotates Left not Right so Left Rot 90 = Right Rotate 270
    rotate = image_one.transpose(PIM.ROTATE_180)
    return rms(rotate, image_two)

def rot90(image_one, image_two):
    # Rotates Left not Right so Left Rot 90 = Right Rotate 270
    rotate = image_one.transpose(PIM.ROTATE_90)
    return diff_finder(rotate, image_two)

def begin(prob):
    _one = prob.figures["1"]
    a = prob.figures["A"]
    _two = prob.figures["2"]
    b = prob.figures["B"]
    _three = prob.figures["3"]
    c = prob.figures["C"]
    _four = prob.figures["4"]
    d = prob.figures["D"]
    _five = prob.figures["5"]
    e = prob.figures["E"]
    _six = prob.figures["6"]
    f = prob.figures["F"]
    _seven = prob.figures["7"]
    g = prob.figures["G"]
    _eight = prob.figures["8"]
    h = prob.figures["H"]
    return [a, b, c, d, e, f, g, h], [_one, _two, _three, _four, _five, _six, _seven, _eight]


def trnsfrmtn(fig1, fig2):
    trans = {'size': comp_size(fig1, fig2), 'dt_fill': dt_fill(fig1, fig2)}
    if inner_shp(fig1, fig2) is not None:
        trans['inner_shape'] = inner_shp(fig1, fig2)
    if outer_shp(fig1, fig2) is not None:
        trans['outer_shape'] = outer_shp(fig1, fig2)
    trans['dt_shpe'] = dt_shpe(fig1, fig2)
    return trans


def gn_tst(init_net, scrs, fgrs, sltns, prob):
    a, c = opn_img(fgrs[0], fgrs[2])
    x = PIC.offset(a, int(a.size[0] / 2), yoffset=0)
    d, f = opn_img(fgrs[3], fgrs[5])
    y = PIC.offset(d, int(d.size[0] / 2), yoffset=0)
    for p, sltn in enumerate(sltns):  # compare init_net with generated sltns
        a_horz = trnsfrmtn(fgrs[6], fgrs[7])
        b_horz = trnsfrmtn(fgrs[7], sltn)
        horz = unn(a_horz, b_horz)
        diag = trnsfrmtn(fgrs[4], sltn)
        a_vert = trnsfrmtn(fgrs[2], fgrs[5])
        b_vert = trnsfrmtn(fgrs[5], sltn)
        vert = unn(a_vert, b_vert)
        score = agnt_comp(init_net, horz, vert, diag)
        q, s = opn_img(fgrs[6], sltns[p])
        z = PIC.offset(q, int(q.size[0] / 2), yoffset=0)
        try:
            if dt_shpe(fgrs[0], fgrs[2]) == 'unchanged' and dt_shpe(fgrs[3], fgrs[5]) == 'unchanged' and -0.011155 * (
                    (fill_amt(fgrs[0]) + fill_amt(fgrs[2])) / 2) + (
                    (fill_amt(fgrs[0]) + fill_amt(fgrs[2])) / 2) <= fill_amt(fgrs[0]) <= (
                    (fill_amt(fgrs[0]) + fill_amt(fgrs[2])) / 2) * 1.011155 and -0.011155 * (
                    (fill_amt(fgrs[3]) + fill_amt(fgrs[5])) / 2) + (
                    (fill_amt(fgrs[3]) + fill_amt(fgrs[5])) / 2) <= fill_amt(fgrs[3]) <= (
                    (fill_amt(fgrs[3]) + fill_amt(fgrs[5])) / 2) * 1.011155 and -0.011155 * (
                    (fill_amt(fgrs[0]) + fill_amt(fgrs[2])) / 2) + (
                    (fill_amt(fgrs[0]) + fill_amt(fgrs[2])) / 2) <= fill_amt(fgrs[2]) <= (
                    (fill_amt(fgrs[0]) + fill_amt(fgrs[2])) / 2) * 1.011155 and -0.011155 * (
                    (fill_amt(fgrs[3]) + fill_amt(fgrs[5])) / 2) + (
                    (fill_amt(fgrs[3]) + fill_amt(fgrs[5])) / 2) <= fill_amt(fgrs[5]) <= (
                    (fill_amt(fgrs[3]) + fill_amt(fgrs[5])) / 2) * 1.011155 and 0.97743902 <= fill_amt(
                sltns[p]) / fill_amt(fgrs[6]) <= 1.2256098 and ((diff_finder(x, PIC.multiply(x,
                                                                                                    c)) < 0.00041 and diff_finder(
                y, PIC.multiply(y, f)) < 0.00041 and diff_finder(z,
                                                                        PIC.multiply(z, s)) < 0.00041) or (
                                                                        translation(fgrs[0], fgrs[
                                                                            2]) <= 0.000243 and translation(fgrs[3],
                                                                                                            fgrs[
                                                                                                                5]) <= 0.000243 and translation(
                                                                    fgrs[6], sltns[p]) <= 0.000243)):
                score += 2.00
            scrs.append(score)
        except:
            scrs.append(score)
    d_check = []
    for i in range(len(scrs)):
        if scrs[i] == max(scrs) or scrs[i] == max(scrs) - 1:
            d_check.append(scrs[i])
    if ot_sh(fgrs[0], fgrs[1]) == True and ot_sh(fgrs[0], fgrs[2]) == True and ot_sh(fgrs[1], fgrs[2]) == True and ot_sh(fgrs[3], fgrs[4]) == True and ot_sh(fgrs[3], fgrs[5]) == True and ot_sh(fgrs[4], fgrs[5]) == True and ot_sh(fgrs[6], fgrs[7]) == True and in_sh(fgrs[0], fgrs[1]) == False and in_sh(fgrs[0], fgrs[2]) == False and in_sh(fgrs[1], fgrs[2]) == False and in_sh(fgrs[3], fgrs[4]) == False and in_sh(fgrs[3], fgrs[5]) == False and in_sh(fgrs[4], fgrs[5]) == False and in_sh(fgrs[6], fgrs[7]) == False:
        arr = []
        for i in range(len(scrs)):
            if (scrs[i] == max(scrs) - 1 or scrs[i] == max(scrs)) and ot_sh(fgrs[6], sltns[i]) == True and ot_sh(fgrs[7], sltns[i]) == True and in_sh(fgrs[6], sltns[i]) == False and in_sh(fgrs[7], sltns[i]) == False:
                arr.append(scrs[i])
            else:
                arr.append(0)
        scrs = arr
    elif ot_sh(fgrs[0], fgrs[3]) == True and ot_sh(fgrs[0], fgrs[6]) == True and ot_sh(fgrs[3], fgrs[6]) == True and ot_sh(fgrs[1], fgrs[4]) == True and ot_sh(fgrs[1], fgrs[7]) == True and ot_sh(fgrs[4], fgrs[7]) == True and ot_sh(fgrs[2], fgrs[5]) == True and in_sh(fgrs[0], fgrs[3]) == False and in_sh(fgrs[0], fgrs[6]) == False and in_sh(fgrs[3], fgrs[6]) == False and in_sh(fgrs[1], fgrs[4]) == False and in_sh(fgrs[1], fgrs[7]) == False and in_sh(fgrs[4], fgrs[7]) == False and in_sh(fgrs[2], fgrs[5]) == False\
            and in_sh(fgrs[0], fgrs[1]) == True and in_sh(fgrs[0], fgrs[2]) == True and in_sh(fgrs[1], fgrs[2]) == True and in_sh(fgrs[3], fgrs[4]) == True and in_sh(fgrs[3], fgrs[5]) == True and in_sh(fgrs[4], fgrs[5]) == True and in_sh(fgrs[6], fgrs[7]) == True:
        arr = []
        for i in range(len(scrs)):
            if scrs[i] == max(scrs) and ot_sh(fgrs[6], sltns[i]) == False and ot_sh(fgrs[7], sltns[i]) == False and in_sh(fgrs[6], sltns[i]) == True and in_sh(fgrs[7], sltns[i]) == True and ot_sh(fgrs[5], sltns[i]) == True:
                arr.append(scrs[i])
            else:
                arr.append(0)
        scrs = arr
    elif 'D' in prob.name.split(' ')[2] and fill_amt(fgrs[0]) < fill_amt(fgrs[5]) < fill_amt(fgrs[7]) and fill_amt(fgrs[4]) < fill_amt(fgrs[6]) < fill_amt(fgrs[2]) and fill_amt(fgrs[1]) < fill_amt(fgrs[3]):
        arr = []
        check = []
        for i in range(len(scrs)):
            if (scrs[i] == max(scrs) - 1) or (scrs[i] == max(scrs)):
                check.append(scrs[i])
        if len(check) > 1:
            for i in range(len(scrs)):
                a_horz = trnsfrmtn(fgrs[6], fgrs[7])
                b_horz = trnsfrmtn(fgrs[7], sltns[i])
                horz = unn(a_horz, b_horz)
                a_vert = trnsfrmtn(fgrs[2], fgrs[5])
                b_vert = trnsfrmtn(fgrs[5], sltns[i])
                vert = unn(a_vert, b_vert)
                if (scrs[i] == max(scrs) - 1 or scrs[i] == max(scrs)) and 'dt_shpe' in horz and 'dt_shpe' in vert:
                    arr.append(scrs[i])
                else:
                    arr.append(0)
        scrs = arr
    elif 'D' in prob.name.split(' ')[2] and 'Challenge' not in prob.name.split(' ')[0] and fill_amt(fgrs[7]) > fill_amt(fgrs[0]) and fill_amt(fgrs[0]) > fill_amt(fgrs[5]) and fill_amt(fgrs[0]) > fill_amt(fgrs[4]) and fill_amt(fgrs[3]) < fill_amt(fgrs[7]) and fill_amt(fgrs[2]) < fill_amt(fgrs[7]) and fill_amt(fgrs[1]) < fill_amt(fgrs[5]) and fill_amt(fgrs[6]) < fill_amt(fgrs[5]):
        arr = []
        check = []
        for i in range(len(scrs)):
            if scrs[i] == max(scrs):
                check.append(scrs[i])
        if len(check) > 1:
            for i in range(len(scrs)):
                diag = trnsfrmtn(fgrs[4], sltns[i])
                if scrs[i] == max(scrs) and diag['dt_fill'] == 'dt_remove' and fill_amt(sltns[i]) > fill_amt(fgrs[6]):
                    arr.append(scrs[i])
                else:
                    arr.append(0)
        scrs = arr
    elif 'D' in prob.name.split(' ')[2] and len(d_check) > 1:
        arr = []
        for i in range(len(scrs)):
            if scrs[i] == max(scrs) - 1:
                arr.append(scrs[i]+1)
            else:
                arr.append(scrs[i])
        scrs = arr
    scrs = norm_scrs(scrs)
    if 1.0 not in scrs:
        m_diagonal = compare_diag(scrs, fgrs, sltns, prob)
        poss_scrs = [m_diagonal]
        m = min(poss_scrs, key=lambda t: t[1])
        scrs = scr_arr(m)
    return scrs


def smntc_net(fgrs):
    b1_h = trnsfrmtn(fgrs[1], fgrs[2])
    a1_h = trnsfrmtn(fgrs[0], fgrs[1])
    a2_h = trnsfrmtn(fgrs[3], fgrs[4])
    b2_h = trnsfrmtn(fgrs[4], fgrs[5])
    a1_v = trnsfrmtn(fgrs[0], fgrs[3])
    b1_v = trnsfrmtn(fgrs[3], fgrs[6])
    a2_v = trnsfrmtn(fgrs[1], fgrs[4])
    b2_v = trnsfrmtn(fgrs[4], fgrs[7])
    return unn(a1_h, b1_h), unn(a2_h, b2_h), unn(a1_v, b1_v), unn(a2_v, b2_v), trnsfrmtn(fgrs[0], fgrs[4])


def agnt_comp(init_net, hor, vert, diag):
    diag1 = init_net[4]
    hrz1 = init_net[0]
    hrz2 = init_net[1]
    vert1 = init_net[2]
    vert2 = init_net[3]
    mtrcs = [wghtd_sim_mtrc(hrz1, hor),
               wghtd_sim_mtrc(hrz2, hor),
               wghtd_sim_mtrc(vert1, vert),
               wghtd_sim_mtrc(vert2, vert),
               wghtd_sim_mtrc(diag1, diag)]
    return float(sum(mtrcs))


def dtr_img_oper(fgrs, ornttn):
    xor_1 = op_trnsfrm(fgrs[0], fgrs[1], 'xor')
    xor_sim_1 = ('xor', rms(xor_1, fgrs[2]))
    intersect_1 = op_trnsfrm(fgrs[0], fgrs[1], 'intersect')
    intersect_sim_1 = ('intersect', rms(intersect_1, fgrs[2]))
    subtract_1 = op_trnsfrm(fgrs[0], fgrs[1], 'subtrct')
    subtract_sim_1 = ('subtrct', rms(subtract_1, fgrs[2]))
    unn_1 = op_trnsfrm(fgrs[0], fgrs[1], 'unn')
    unn_sim_1 = ('unn', rms(unn_1, fgrs[2]))
    mod_subtract_1 = op_trnsfrm(fgrs[0], fgrs[1], 'mod-subtrct-' + ornttn)
    mod_subtract_sim_1 = ('mod-subtrct-' + ornttn, rms(mod_subtract_1, fgrs[2]))
    compares = [xor_sim_1, intersect_sim_1, unn_sim_1, subtract_sim_1, mod_subtract_sim_1]
    return min(compares, key=lambda t: t[1])[0]


def dtr_single_img_op(fgrs):
    a0b1c2 = dtr_img_oper([fgrs[0], fgrs[1], fgrs[2]], 'horiz')
    a0d3g6 = dtr_img_oper([fgrs[0], fgrs[3], fgrs[6]], 'vert')
    d3e4f5 = dtr_img_oper([fgrs[3], fgrs[4], fgrs[5]], 'horiz')
    b1e4h7 = dtr_img_oper([fgrs[1], fgrs[4], fgrs[7]], 'vert')
    ops = [a0b1c2, a0d3g6, d3e4f5, b1e4h7]
    return max(set(ops), key=ops.count)


def img_op_solver(fgrs, sltns):
    oprtn = dtr_single_img_op(fgrs)
    horz_tester = op_trnsfrm(fgrs[6], fgrs[7], oprtn)
    vert_tester = op_trnsfrm(fgrs[2], fgrs[5], oprtn)
    horz_scrs = []
    vert_scrs = []
    for p, sltn in enumerate(sltns):
        x = rms(horz_tester, sltn)
        horz_scrs.append((p, x))
    for p, sltn in enumerate(sltns):
        x = rms(vert_tester, sltn)
        vert_scrs.append((p, x))
    m_horz = min(horz_scrs, key=lambda t: t[1])
    m_vert = min(vert_scrs, key=lambda t: t[1])
    if m_horz[0] == m_vert[0] or oprtn == 'mod-subtrct-horiz':
        if m_horz[0] == m_vert[0] and (fill_amt(fgrs[2]) - 6 <= fill_amt(fgrs[0]) - fill_amt(fgrs[1]) <= fill_amt(fgrs[2]) + 6) and (fill_amt(fgrs[5]) - 6 <= fill_amt(fgrs[3]) - fill_amt(fgrs[4]) <= fill_amt(fgrs[5]) + 6):
            arr = []
            for i in range(len(horz_scrs)):
                if fill_amt(sltns[i]) - 6 <= fill_amt(fgrs[6]) - fill_amt(fgrs[7]) <= fill_amt(fgrs[i]) + 6:
                    arr.append(horz_scrs[i])
            m_horz = min(arr, key=lambda t: t[1])
            scrs = scr_arr(m_horz)
            return scrs
        else:
            scrs = scr_arr(m_horz)
            return scrs
    else:
        scrs = scr_arr(m_vert)
        return scrs


def dct_comp(a1, a2):
    a2_kys = set(a2.keys())
    a1_kys = set(a1.keys())
    intersect_kys = a1_kys.intersection(a2_kys)
    remove = a2_kys - a1_kys
    add = a1_kys - a2_kys
    same = set(i for i in intersect_kys if a1[i] == a2[i])
    mod = {i: (a1[i], a2[i]) for i in intersect_kys if a1[i] != a2[i]}
    return add, remove, mod, same


def unn(trns1, trns2):
    add, remove, mod, same = dct_comp(trns1, trns2)
    d_unn = {}
    for key in same:
        d_unn[key] = trns1[key]
    return d_unn


def wghtd_sim_mtrc(a, b):  # calcs weights based on how many similar items there are
    add, remove, mod, same = dct_comp(a, b)
    scr = 0
    # noinspection PyUnusedLocal
    for i in same:
        scr += 1
    return scr


def scr_arr(a):  # Sets figure that has the best score to 1
    scrs = [0, 0, 0, 0, 0, 0, 0, 0]
    scrs[a[0]] = 1
    return scrs


def norm_scrs(scrs):
    if sum(scrs) == 0:
        normalized = [.1250, .1250, .1250, .1250, .1250, .1250, .1250, .1250]
    else:
        mx_sc = max(scrs)
        for i, score in enumerate(scrs):
            if score != mx_sc:
                scrs[i] = 0
        t = float(sum(scrs))
        normalized = [x / t for x in scrs]
    return normalized


def opn_img(img1, img2):
    if hasattr(img1, 'visualFilename'):
        src = PIM.open(img1.visualFilename)
    else:
        src = img1
    if hasattr(img2, 'visualFilename'):
        comp = PIM.open(img2.visualFilename)
    else:
        comp = img2
    return src, comp


def blk_pxl_array_conv(pxls):
    nds = []
    for hor in range(184):
        for vert in range(184):
            if pxls[hor, vert] == (0, 0, 0, 255):  # black
                nds.append((hor, vert))
    return nds


def clsst_nd(nd, nds):
    try:
        nds = num.asarray(nds)
        dis2 = num.sum((nds - nd) ** 2, axis=1)
        ans = num.argmin(dis2)
    except:
        ans = 0
    return ans


def eql(src, comp):
    if round(rms(src, comp), 0) < 970.0:
        return True
    else:
        return False


def op_trnsfrm(img1, img2, oprtn):
    source, compare = opn_img(img1, img2)
    if oprtn == 'mod-subtrct-vert':
        return mod_sub_minus(source, compare, 'vert')
    elif oprtn == 'xor':
        return xor_func(source, compare)
    elif oprtn == 'unn':
        return PIC.multiply(source, compare)
    elif oprtn == 'subtrct':
        return sub_minus(source, compare)
    elif oprtn == 'intersect':
        return intrsct(source, compare)
    elif oprtn == 'mod-subtrct-horiz':
        return mod_sub_minus(source, compare, 'horiz')


def outer_shp(src, comp):
    src_shpe = shpe_gt(rgn_finder(src))
    comp_shpe = shpe_gt(rgn_finder(comp))
    if len(comp_shpe) < 2 or len(src_shpe) < 2:
        return None
    else:
        src_inner = center_gt(src_shpe)
        comp_inner = center_gt(comp_shpe)
        src_shpe.pop(src_inner)
        comp_shpe.pop(comp_inner)
        src_outer = write_shpe(src_shpe)
        comp_outer = write_shpe(comp_shpe)
        return eql(src_outer, comp_outer)

def ot_sh(src, comp):
    src_shpe = shpe_gt(rgn_finder(src))
    comp_shpe = shpe_gt(rgn_finder(comp))
    if len(comp_shpe) < 2 or len(src_shpe) < 2:
        return None
    else:
        src_inner = center_gt(src_shpe)
        comp_inner = center_gt(comp_shpe)
        src_shpe.pop(src_inner)
        comp_shpe.pop(comp_inner)
        src_outer = write_shpe(src_shpe)
        comp_outer = write_shpe(comp_shpe)
    if round(rms(src_outer, comp_outer)) <= 963.0:
        return True
    else:
        return False


def strict_eql(src, comp):
    if round(rms(src, comp), 0) < 960.0:
        return True
    else:
        return False


def inner_shp(src, comp):
    src_shpe = shpe_gt(rgn_finder(src))
    comp_shpe = shpe_gt(rgn_finder(comp))
    try:
        src_inner = src_shpe[center_gt(src_shpe)]
    except:
        src_inner = src
    try:
        comp_inner = comp_shpe[center_gt(comp_shpe)]
    except:
        comp_inner = comp
    return eql(src_inner, comp_inner)


def in_sh(src, comp):
    src_shpe = shpe_gt(rgn_finder(src))
    comp_shpe = shpe_gt(rgn_finder(comp))
    try:
        src_inner = src_shpe[center_gt(src_shpe)]
    except:
        src_inner = src
    try:
        comp_inner = comp_shpe[center_gt(comp_shpe)]
    except:
        comp_inner = comp
    if round(rms(src_inner, comp_inner)) <= 960.0:
        return True
    else:
        return False

def in_sh_rms(src, comp):
    src_shpe = shpe_gt(rgn_finder(src))
    comp_shpe = shpe_gt(rgn_finder(comp))
    try:
        src_inner = src_shpe[center_gt(src_shpe)]
    except:
        src_inner = src
    try:
        comp_inner = comp_shpe[center_gt(comp_shpe)]
    except:
        comp_inner = comp
    return rms(src_inner, comp_inner)


def comp_size(src, comp):
    comp_sz = shpe_size(comp)
    src_size = shpe_size(src)
    if src_size > comp_sz and (src_size - comp_sz) > 1000:
        return 'contracted'
    elif comp_sz > src_size and (comp_sz - src_size) > 1000:
        return 'expanded'
    else:
        return 'unchanged'


def dt_fill(src, comp):
    cnt_comp = fill_amt(comp)
    cnt_src = fill_amt(src)
    if cnt_src > cnt_comp and (cnt_src - cnt_comp) > 1500:
        return 'dt_remove'
    elif cnt_src < cnt_comp and (cnt_comp - cnt_src) > 1500:
        return 'dt_add'
    else:
        return 'unchanged'


def dt_shpe(src, comp):
    cnt_comp = len(rgn_finder(comp))
    cnt_src = len(rgn_finder(src))
    if cnt_src > cnt_comp:
        return 'dt_remove'
    elif cnt_comp > cnt_src:
        return 'dt_add'
    else:
        return 'unchanged'


def comp_adcf(fgrs):
    shp0a = shpe_gt(rgn_finder(fgrs[0]))
    shp3d = shpe_gt(rgn_finder(fgrs[3]))
    shp2c = shpe_gt(rgn_finder(fgrs[2]))
    shp5f = shpe_gt(rgn_finder(fgrs[5]))
    if len(shp0a) < 2 or len(shp2c) < 2 or len(shp3d) < 2 or len(shp5f) < 2:
        return False
    if strict_eql(shp0a[top_gt(shp0a)], shp2c[top_gt(shp2c)]) \
            and strict_eql(shp3d[top_gt(shp3d)], shp5f[top_gt(shp5f)]):
        return True
    else:
        return False


def comp_bcef(fgrs):
    shp1b = shpe_gt(rgn_finder(fgrs[1]))
    shp2c = shpe_gt(rgn_finder(fgrs[2]))
    shp4e = shpe_gt(rgn_finder(fgrs[4]))
    shp5f = shpe_gt(rgn_finder(fgrs[5]))
    if len(shp1b) < 2 or len(shp2c) < 2 or len(shp4e) < 2 or len(shp5f) < 2:
        return False
    if strict_eql(shp1b[bttm_gt(shp1b)], shp2c[bttm_gt(shp2c)]) \
            and strict_eql(shp4e[bttm_gt(shp4e)], shp5f[bttm_gt(shp5f)]):
        return True
    else:
        return False


def top_comp_bttm(scrs, fgrs, sltns):
    if not scrs:
        scrs = [.1250, .1250, .1250, .1250, .1250, .1250, .1250, .1250]
    g_shpe = shpe_gt(rgn_finder(fgrs[6]))
    h_shpe = shpe_gt(rgn_finder(fgrs[7]))
    top_g = g_shpe[top_gt(g_shpe)]
    bottom_h = h_shpe[bttm_gt(h_shpe)]
    poss_ans = []
    comp2 = []
    for p, scr in enumerate(scrs):
        if scr != 0.0:
            poss_ans.append((p, sltns[p]))
    comps = []
    for ans in poss_ans:
        solution_shpe = shpe_gt(rgn_finder(ans[1]))
        try:
            tp_sltn = solution_shpe[top_gt(solution_shpe)]
        except:
            tp_sltn = top_g
        try:
            bttm_sltn = solution_shpe[bttm_gt(solution_shpe)]
        except:
            bttm_sltn = bottom_h
        x = (ans[0], rms(top_g, tp_sltn) + rms(bottom_h, bttm_sltn))
        comps.append(x)
    for i in comps:
        if fill_amt(fgrs[6]) < fill_amt(fgrs[7]) < fill_amt(sltns[i[0]]) and fill_amt(fgrs[5]) > fill_amt(
                fgrs[2]) > fill_amt(fgrs[1]) > fill_amt(fgrs[0]):
            comp2.append(i)
    if not comp2:
        comp2 = comps
    return min(comp2, key=lambda t: t[1])


def nearest(arr, val, index):
    fill_items = []
    for i in arr:
        fill_items.append(i[index])
    arr = num.asarray(fill_items)
    return (num.abs(arr - val)).argmin()


def compare_diag(scrs, fgrs, sltns, prob):
    if not scrs:
        scrs = [.1250, .1250, .1250, .1250, .1250, .1250, .1250, .1250]
    possible_answers = []
    for i, score in enumerate(scrs):
        if score != 0.0:
            possible_answers.append((i, sltns[i]))
    comparisons = []
    for answer in possible_answers:
        x = (answer[0], rms(fgrs[4], answer[1]))
        comparisons.append(x)
    comp2 = []
    comp_temp = []
    avg_cd = 0.000
    rms_arr = []
    fill = []
    item = 'all'
    b, d = opn_img(fgrs[1], fgrs[3])
    s_comp = PIC.darker(b, d)
    c, g = opn_img(fgrs[2], fgrs[6])
    e_comp = PIC.darker(c, g)
    for i in comparisons:  # loop through options with close RMSes
        a, f = opn_img(fgrs[0], fgrs[5])
        h_comp = PIC.darker(a, f)
        s, e = opn_img(sltns[i[0]], fgrs[4])
        c_comp = PIC.darker(g, e)
        if fill_amt(fgrs[0]) == fill_amt(fgrs[2]) and fill_amt(fgrs[3]) == fill_amt(fgrs[5]) and fill_amt(
                fgrs[2]) != fill_amt(fgrs[5]) and fill_amt(fgrs[1]) != fill_amt(fgrs[2]):
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[6]) == fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
        elif fill_amt(fgrs[0]) < fill_amt(fgrs[4]) and fill_amt(fgrs[3]) < fill_amt(fgrs[7]) and fill_amt(
                fgrs[2]) < fill_amt(fgrs[5]) and fill_amt(fgrs[6]) < fill_amt(fgrs[7]):
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[4]) < fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
            if fill_amt(fgrs[0]) > 0 and fill_amt(fgrs[3]) > 0:
                avg_cd = (fill_amt(fgrs[4]) / fill_amt(fgrs[0]) + fill_amt(fgrs[7]) / fill_amt(fgrs[3])) / 2
            elif fill_amt(fgrs[0]) > 0 and fill_amt(fgrs[3]) == 0:
                avg_cd = fill_amt(fgrs[4]) / fill_amt(fgrs[0])
            elif fill_amt(fgrs[0]) == 0 and fill_amt(fgrs[3]) > 0:
                avg_cd = fill_amt(fgrs[7]) / fill_amt(fgrs[3])
        elif fill_amt(fgrs[0]) > fill_amt(fgrs[4]) and fill_amt(fgrs[3]) > fill_amt(fgrs[7]) and fill_amt(
                fgrs[2]) > fill_amt(fgrs[5]) and fill_amt(fgrs[6]) > fill_amt(fgrs[7]) and 'D' not in prob.name.split(' ')[2]:
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[4]) > fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
            if fill_amt(fgrs[0]) > 0 and fill_amt(fgrs[3]) > 0:
                avg_cd = (fill_amt(fgrs[4]) / fill_amt(fgrs[0]) + fill_amt(fgrs[7]) / fill_amt(fgrs[3])) / 2
            elif fill_amt(fgrs[0]) > 0 and fill_amt(fgrs[3]) == 0:
                avg_cd = fill_amt(fgrs[4]) / fill_amt(fgrs[0])
            elif fill_amt(fgrs[0]) == 0 and fill_amt(fgrs[3]) > 0:
                avg_cd = fill_amt(fgrs[7]) / fill_amt(fgrs[3])
        elif fill_amt(fgrs[2]) > fill_amt(fgrs[5]) and fill_amt(fgrs[6]) > fill_amt(fgrs[7]) and 'D' not in prob.name.split(' ')[2]:
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[4]) > fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
            if fill_amt(fgrs[2]) > 0 and fill_amt(fgrs[6]) > 0:
                avg_cd = (fill_amt(fgrs[5]) / fill_amt(fgrs[2]) + fill_amt(fgrs[7]) / fill_amt(fgrs[6])) / 2
            elif fill_amt(fgrs[2]) > 0 and fill_amt(fgrs[6]) == 0:
                avg_cd = fill_amt(fgrs[5]) / fill_amt(fgrs[2])
            elif fill_amt(fgrs[2]) == 0 and fill_amt(fgrs[6]) > 0:
                avg_cd = fill_amt(fgrs[7]) / fill_amt(fgrs[6])
        elif 'D' not in prob.name.split(' ')[2] and fill_amt(fgrs[2]) < fill_amt(fgrs[5]) and fill_amt(fgrs[6]) < fill_amt(fgrs[7]):
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[4]) < fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
            if fill_amt(fgrs[2]) > 0 and fill_amt(fgrs[6]) > 0:
                avg_cd = (fill_amt(fgrs[5]) / fill_amt(fgrs[2]) + fill_amt(fgrs[7]) / fill_amt(fgrs[6])) / 2
            elif fill_amt(fgrs[2]) > 0 and fill_amt(fgrs[6]) == 0:
                avg_cd = fill_amt(fgrs[5]) / fill_amt(fgrs[2])
            elif fill_amt(fgrs[2]) == 0 and fill_amt(fgrs[6]) > 0:
                avg_cd = fill_amt(fgrs[7]) / fill_amt(fgrs[6])
        elif 'D' not in prob.name.split(' ')[2] and fill_amt(fgrs[0]) < fill_amt(fgrs[4]) and fill_amt(fgrs[3]) < fill_amt(fgrs[7]):
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[4]) < fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
            if fill_amt(fgrs[0]) > 0 and fill_amt(fgrs[3]) > 0:
                avg_cd = (fill_amt(fgrs[4]) / fill_amt(fgrs[0]) + fill_amt(fgrs[7]) / fill_amt(fgrs[3])) / 2
            elif fill_amt(fgrs[0]) > 0 and fill_amt(fgrs[3]) == 0:
                avg_cd = fill_amt(fgrs[4]) / fill_amt(fgrs[0])
            elif fill_amt(fgrs[0]) == 0 and fill_amt(fgrs[3]) > 0:
                avg_cd = fill_amt(fgrs[7]) / fill_amt(fgrs[3])
        elif 'D' not in prob.name.split(' ')[2] and fill_amt(fgrs[0]) > fill_amt(fgrs[4]) and fill_amt(fgrs[3]) > fill_amt(fgrs[7]):
            fr = fill_amt(sltns[i[0]])
            if fill_amt(fgrs[4]) > fr:
                z = (i[0], i[1], fr)
                comp2.append(z)
        else:
            if 'D' not in prob.name.split(' ')[2]:
                fr = fill_amt(sltns[i[0]])
                z = (i[0], i[1], fr)
                comp2.append(z)
                item = 'seven'
            else:
                if rms(fgrs[2],fgrs[4]) <= 960.0 and rms(fgrs[4], fgrs[6]) <= 960.0 and rms(fgrs[1], fgrs[3]) <= 960.0 and rms(fgrs[5], fgrs[7]) <= 960.0 and eql(fgrs[1], fgrs[5]) == False and eql(fgrs[0], fgrs[5]) == False:
                    a, b = opn_img(fgrs[2], sltns[6])
                    if rms(a.transpose(PIM.FLIP_LEFT_RIGHT),sltns[i[0]]) > 961.0 and eql(sltns[i[0]],fgrs[0]) == False and eql(sltns[i[0]],fgrs[6]) == False and eql(sltns[i[0]],fgrs[7]) == False and eql(sltns[i[0]],fgrs[5]) == False and eql(sltns[i[0]],fgrs[2]) == False and eql(sltns[i[0]],fgrs[1]) == False:
                        comp2.append(i)
                    item = 'eight'
                elif fill_amt(fgrs[2]) > fill_amt(fgrs[1]) > fill_amt(fgrs[0]) and fill_amt(fgrs[6]) > fill_amt(fgrs[3]) > fill_amt(fgrs[0]) and fill_amt(sltns[i[0]]) > fill_amt(fgrs[4]) > fill_amt(fgrs[0]) and fill_amt(fgrs[7]) > fill_amt(fgrs[4]) and fill_amt(fgrs[5]) > fill_amt(fgrs[4]):
                    a, b = opn_img(fgrs[5], fgrs[7])
                    out = PIC.darker(a, b)
                    rms_2 = rms(out, sltns[i[0]])
                    fr = fill_amt(sltns[i[0]])
                    z = (i[0], i[1], fr, rms_2)
                    comp2.append(z)
                    item = 'nine'
                elif rms(e_comp, fgrs[4]) <= 962.0 and rms(s_comp, sltns[i[0]]) <= 962.0 and fill_amt(fgrs[0]) > fill_amt(fgrs[2]) > fill_amt(fgrs[1]) and fill_amt(fgrs[4]) > fill_amt(fgrs[5]) > fill_amt(fgrs[3]) and fill_amt(fgrs[0]) - 1774 > fill_amt(fgrs[5]) and fill_amt(fgrs[0]) - 1774 > fill_amt(fgrs[7]) and fill_amt(fgrs[4]) - 1774 > fill_amt(fgrs[2]) and fill_amt(fgrs[4]) - 1774 > fill_amt(fgrs[6]) and fill_amt(sltns[i[0]]) - 1774 > fill_amt(fgrs[1]) and fill_amt(sltns[i[0]]) - 1774 > fill_amt(fgrs[3]):
                    item = 'ten'
                    comp2.append(i)
                elif rms(h_comp, fgrs[7]) <= 962.0 and rms(c_comp, fgrs[2]) <= 962.0 and rms(fgrs[3], PIC.darker(b, s)) <= 962.0:
                    item = 'ten'
                    comp2.append(i)
                elif rot180_d(fgrs[0], fgrs[2]) <= 963.0 and rot180_d(fgrs[3], fgrs[5]) <= 963.0 and rot180_d(fgrs[6], sltns[i[0]]) <= 963.0:
                    comp2.append(i)
                    item = 'eleven'
                elif 'Challenge' not in prob.name.split(' ')[0] and 'Basic' not in prob.name.split(' ')[0] and not ('01' in prob.name.split(' ')[2] or '02' in prob.name.split(' ')[2] or '03' in prob.name.split(' ')[2] or '04' in prob.name.split(' ')[2] or '06' in prob.name.split(' ')[2] or '11' in prob.name.split(' ')[2]):
                    d_guess = 0
    if item == 'seven':
        for i in comp2:
            rms_arr.append(i[1])
            fill.append(i[1])
    if item == 'seven' and len(comp2) > 1 and len(set(rms_arr)) == len(set(fill)) and len(set(fill)) < len(comp2):
        b, h = opn_img(fgrs[1], fgrs[7])
        d, f = opn_img(fgrs[3], fgrs[5])
        g, c = opn_img(fgrs[2], fgrs[6])
        avg_cd = (rot180_v2(b, h) + rot180_v2(d, f) + rot180_v2(g, c)) / 3
        for i in comp2:
            a, x = opn_img(fgrs[0], sltns[i[0]])
            comp_temp.append((i[0], i[1], i[2], rot180_v2(a, x)))
    try:
        if avg_cd != 0 and item != 'seven':
            if ('C' in prob.name.split(' ')[2] and 'B' in prob.name.split(' ')[0]) or (
                    'D' in prob.name.split(' ')[2] and 'C' in prob.name.split(' ')[0]):
                ans = min(comp2, key=lambda t: t[1])
            else:
                comp3 = comp2[nearest(comp2, avg_cd * fill_amt(fgrs[4]), 2)]
                ans = comp3
        elif item == 'seven':
            comp3 = comp_temp[nearest(comp_temp, avg_cd, 3)]
            ans = comp3
        elif item == 'eight' or item == 'ten' or item == 'eleven':
            ans = min(comp2, key=lambda t: t[1])
        elif item == 'nine':
            ans = min(comp2, key=lambda t: t[3])
        elif item == 'd_guess':
            ans = d_guess
        else:
            ans = min(comparisons, key=lambda t: t[1])
    except:
        ans = min(comparisons, key=lambda t: t[1])
    return ans


def equal_imgs(image_one, image_two):
    if num.array_equal(image_one, image_two):
        return True
    else:
        return False


def rot180_v2(image_one, image_two):
    rotate = image_two.transpose(PIM.ROTATE_180)
    return rms(image_one, rotate)


def diff_finder(image_one, image_two):
    pr = zip(image_one.getdata(), image_two.getdata())
    if len(image_one.getbands()) == 1:
        difference = sum(abs(-pr1 + pr0) for pr0, pr1 in pr)
    else:
        difference = sum(abs(-qr1 + qr0) for pr0, pr1 in pr for qr0, qr1 in zip(pr0, pr1))
    element_cnt = 3.00 * image_one.size[0] * image_one.size[1]
    return (difference / 25500.00) / element_cnt


class Rgn:
    def __init__(self, h, v):
        self._pixels = [(h, v)]
        self._mn_h = h
        self._mx_h = h
        self._mn_v = v
        self._mx_v = v

    def addition(self, h, v):
        self._pixels.append((h, v))
        self._mn_h = min(self._mn_h, h)
        self._mx_h = max(self._mx_h, h)
        self._mn_v = min(self._mn_v, v)
        self._mx_v = max(self._mx_v, v)


def shpe_size(fig):
    img = PIM.open(fig.visualFilename)
    wdh, hgt = img.size
    pxls = img.load()
    blck_pxls = []
    for x in range(wdh):
        for y in range(hgt):
            if pxls[x, y] == (0, 0, 0, 255):  # RGBA - (0, 0, 0, 255) is a black pixel
                blck_pxls.append((x, y))
    try:
        x_blck = min(blck_pxls, key=lambda z: (z[1]))[1], max(blck_pxls, key=lambda q: (q[1]))[1]
        y_blck = min(blck_pxls)[0], max(blck_pxls)[0]
        b_wdh = x_blck[1] - x_blck[0]
        b_hgt = y_blck[1] - y_blck[0]
        dim = b_wdh * b_hgt
    except:
        dim = 0
    return dim


def center_gt(shpe):
    closest = []
    for blb in shpe:
        try:
            nds = blk_pxl_array_conv(blb.load())
            closest.append(nds[clsst_nd((92, 92), nds)])
        except:
            continue
    return clsst_nd((92, 92), closest)


def top_gt(shpe):
    closest = []
    for blb in shpe:
        nds = blk_pxl_array_conv(blb.load())
        closest.append(nds[clsst_nd((92, 92), nds)])
    return clsst_nd((0, 0), closest)


def bttm_gt(shpe):
    closest = []
    for blb in shpe:
        nds = blk_pxl_array_conv(blb.load())
        closest.append(nds[clsst_nd((92, 92), nds)])
    return clsst_nd((0, 184), closest)

def rot180(image_one, image_two):
    rotate = image_two.transpose(PIM.ROTATE_180)
    return diff_finder(image_one, rotate)

def write_shpe(shpe):
    otr_shps = PIM.new("RGBA", (184, 184), "white")
    otr_pxls = otr_shps.load()
    for blb in shpe:
        pxls = blb.load()
        for hor in range(184):  # find rgns 1st pass
            for vert in range(184):
                if (0, 0, 0, 255) == pxls[hor, vert]:
                    otr_pxls[hor, vert] = (0, 0, 0, 255)
    return otr_shps


def rgn_finder(fig):
    if isinstance(fig, dict):
        img = PIM.open(fig['visualFilename'])
    else:
        img = PIM.open(fig.visualFilename)
    wdh, hgt = img.size
    n_rgns = 0
    pxls = img.load()
    eqvlncs = {}
    rgns = {}
    # noinspection PyUnusedLocal
    pxl_rgn = [[0 for y in range(hgt)] for x in range(wdh)]
    for h in range(hgt):
        for w in range(wdh): # switched w and h
            if (0, 0, 0, 255) == pxls[w, h]:  # Gets rgn no. from N or W or creates new rgn
                w_rgn = pxl_rgn[w][h - 1] if h > 0 else 0
                n_rgn = pxl_rgn[w - 1][h] if w > 0 else 0
                max_rgn = max(n_rgn, w_rgn)
                if max_rgn > 0:  # Nghbr alrdy has rgn; new rgn is smallest > 0
                    nw_rgn = min(filter(lambda i: i > 0, (n_rgn, w_rgn)))
                    if max_rgn in eqvlncs and max_rgn > nw_rgn:  # update eqvlncs
                        eqvlncs[max_rgn].add(nw_rgn)
                    elif max_rgn > nw_rgn:
                        eqvlncs[max_rgn] = {nw_rgn}
                else:
                    n_rgns += 1
                    nw_rgn = n_rgns
                pxl_rgn[w][h] = nw_rgn
    for v in range(wdh):  # Rescan img giving equal rgns same value
        for h in range(hgt):
            r = pxl_rgn[v][h]
            if r > 0:
                while r in eqvlncs:
                    r = min(eqvlncs[r])
                if r not in rgns:
                    rgns[r] = Rgn(v, h)
                else:
                    rgns[r].addition(v, h)
    return list(rgns.items())


def shpe_gt(rgns):
    shpe = []
    for rgn in rgns:
        pxls = rgn[1]._pixels
        blb = PIM.new("RGBA", (184, 184), "white")
        blb_pxls = blb.load()
        for p in pxls:
            blb_pxls[p[0], p[1]] = (0, 0, 0, 255)
        shpe.append(blb)
    return shpe


def rms(img1, img2):
    src, comp = opn_img(img1, img2)
    df = PIC.difference(src, comp).histogram()
    sq = (value * (idx ** 2) for idx, value in enumerate(df))
    return round((((sum(sq)) / float(src.size[0] * src.size[1])) ** 0.5), 0)


def sub_minus(src, comp):
    comp_pxls = comp.load()
    wdh, hgt = src.size
    src_pxls = src.load()
    for x in range(wdh):
        for y in range(hgt):
            if comp_pxls[x, y] == (0, 0, 0, 255):
                src_pxls[x, y] = (255, 255, 255, 255)
                src_pxls[x + 1, y] = (255, 255, 255, 255)
                src_pxls[x, y + 1] = (255, 255, 255, 255)
                src_pxls[x - 1, y] = (255, 255, 255, 255)
                src_pxls[x, y - 1] = (255, 255, 255, 255)
    return src


def rot270(image_one, image_two):
    # Rotates Left not Right so Left Rot 90 = Right Rotate 270
    rotate = image_one.transpose(PIM.ROTATE_270)
    return diff_finder(rotate, image_two)


def intrsct(srce, comp):
    wdh, hgt = 184, 184
    compare_pxls = comp.load()
    source_pxls = srce.load()
    intersect_image = PIM.new("RGBA", (wdh, hgt), "white")
    intersect_pixels = intersect_image.load()
    for x in range(wdh):
        for y in range(hgt):
            if compare_pxls[x, y] == (0, 0, 0, 255) and source_pxls[x, y] == (0, 0, 0, 255):
                intersect_pixels[x, y] = (0, 0, 0, 255)
    return intersect_image


def mod_sub_minus(src, comp, orient):
    wdh, hgt = 184, 184
    compare_pixels = comp.load()
    source_pixels = src.load()
    for x in range(wdh):
        for y in range(hgt):
            if compare_pixels[x, y] == (0, 0, 0, 255):
                if 51 <= x and orient == 'horiz':
                    source_pixels[x - 51, y] = (255, 255, 255, 255)
                elif (184 - 51) > y and orient == 'vert':
                    source_pixels[x, y + 51] = (255, 255, 255, 255)
    mod_img = PIM.new("RGBA", (wdh, hgt), "white")
    mod_pxls = mod_img.load()
    for x in range(wdh):
        for y in range(hgt):
            if source_pixels[x, y] == (0, 0, 0, 255):
                if 20 <= x and orient == 'horiz':
                    mod_pxls[x - 20, y] = (0, 0, 0, 255)
                elif (184 - 20) > y and orient == 'vert':
                    mod_pxls[x, y + 20] = (0, 0, 0, 255)
    return mod_img


def xor_func(src, comp):
    wdh, hgt = src.size
    src_pxls = src.load()
    comp_pxls = comp.load()
    xor_image = PIM.new("RGBA", (wdh, hgt), "white")
    xor_pxls = xor_image.load()
    for x in range(wdh):
        for y in range(hgt):
            if (src_pxls[x, y] == (0, 0, 0, 255) and comp_pxls[x, y] != (0, 0, 0, 255)
                    or src_pxls[x, y] != (0, 0, 0, 255) and comp_pxls[x, y] == (0, 0, 0, 255)):
                xor_pxls[x, y] = (0, 0, 0, 255)
    return xor_image


def translation(src, comp):
    comp = PIM.open(comp.visualFilename)
    src_shpe = shpe_gt(rgn_finder(src))
    try:
        src_inner = src_shpe[center_gt(src_shpe)]
    except:
        src_inner = src
    if len(src_shpe) < 2:
        return None
    else:
        src_in = center_gt(src_shpe)
        src_shpe.pop(src_in)
        src_outer = write_shpe(src_shpe)
        x = PIC.offset(src_inner, int((.95 * src_inner.size[0] / 5) - 2), yoffset=0)
        y = PIC.offset(src_outer, int((11 * src_outer.size[0] / 16) - 2), yoffset=0)
        z = PIC.multiply(x, y)
        return diff_finder(z, comp)


def get_imgs(prob, name):
    fig = prob.figures[name]
    img = PIM.open(fig.visualFilename).convert('L')
    return num.array(img)


def refTB(image_one, image_two):
    flip = image_one.transpose(PIM.FLIP_TOP_BOTTOM)
    return diff_finder(flip, image_two)


def diss_obj(image_one, image_two):
    df = diff_finder(image_one, image_two)
    if df != 0.0:
        return True


def ff_check(image_one, image_two):
    return diff_finder(image_one, image_two)


def refLR(image_one, image_two):
    ref = image_one.transpose(PIM.FLIP_LEFT_RIGHT)
    return diff_finder(ref, image_two)


def drk_pxl_perc(image_one_arr, image_two_arr):
    im2 = (float(image_two_arr.size) - float(num.count_nonzero(image_two_arr))) / float(image_two_arr.size)
    im1 = (float(image_one_arr.size) - float(num.count_nonzero(image_one_arr))) / float(image_one_arr.size)
    return (im2 - im1) / im1


class Agent:
    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def Solve(self, problem):
        ans = 3  # Guess instead of -1
        if problem.problemType == '2x2':
            c_image = PIM.open(problem.figures['C'].visualFilename).convert('L')
            c_image_arr = get_imgs(problem, 'C')
            b_image = PIM.open(problem.figures['B'].visualFilename).convert('L')
            b_image_arr = get_imgs(problem, 'B')
            a_image = PIM.open(problem.figures['A'].visualFilename).convert('L')
            a_image_arr = get_imgs(problem, 'A')
            if equal_imgs(a_image_arr, b_image_arr):
                for i in sorted(problem.figures):
                    if i.isnumeric() and equal_imgs(c_image_arr, get_imgs(problem, i)):
                        ans = int(i)
            elif refLR(a_image, b_image) < 0.00001:  # Rounding without the Math Package
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = refLR(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + diff.index(min(diff)))
            elif rot270(a_image, b_image) < 0.00005:  # Rounding without the Math Package
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = rot270(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + diff.index(min(diff)))
            elif refTB(a_image, c_image) < 0.00001:  # Rounding without the Math Package
                diff = []
                drk = []
                final = []
                inx = []
                top = drk_pxl_perc(a_image_arr, b_image_arr)
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = refTB(b_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        dinstance = 0.0 if top == 0 else (drk_pxl_perc(c_image_arr, get_imgs(problem, i)) - top) / top
                        diff.append(instance)
                        drk.append(dinstance)
                for d in drk:
                    if -0.02 <= d <= 0.02:  # within 20% change of the top change
                        index = int(drk.index(d))
                        inx.append(index)
                        final.append(diff[index])
                try:
                    ans = int(1 + inx[final.index(min(final))])
                except Exception:
                    ans = int(1 + diff.index(min(diff)))
            elif 12 < ff_check(a_image, b_image) < 15:
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = ff_check(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + diff.index(max(diff)))
            elif rot270(a_image, b_image) < 0.0003 and rot270(a_image, b_image) < diff_finder(a_image, b_image) and rot270(a_image, b_image) < refLR(a_image, b_image) and rot270(a_image, b_image) < refTB(a_image, c_image) and rot270(a_image, b_image) < rot90(a_image, b_image):
                diff = []
                top = rot270(a_image, b_image)
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = rot270(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        inst = 0.0 if top == 0 else (instance - top) / top
                        diff.append(inst)
                ans = int(1 + diff.index(min(diff, key=abs)))
            elif diff_finder(a_image, b_image) > 1.0 > refLR(a_image, b_image):
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = refLR(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + diff.index(min(diff)))
            elif 1.0 > refTB(a_image, c_image) or 1.0 > rot90(a_image, b_image):
                diff = []
                rdiff = []
                drk = []
                final = []
                inx = []
                top = drk_pxl_perc(a_image_arr, b_image_arr)
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = refTB(b_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        rinstance = rot90(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        dinstance = 0.0 if top == 0 else (drk_pxl_perc(c_image_arr, get_imgs(problem, i)) - top) / top
                        diff.append(instance)
                        rdiff.append(rinstance)
                        drk.append(dinstance)
                if diff and rdiff and min(diff) < min(rdiff) + 0.00029:  # standard deviation / error
                    for i in drk:
                        if -0.2 <= i <= 0.2:  # within 20% change of the top change
                            index = int(drk.index(i))
                            inx.append(index)
                            final.append(diff[index])
                    try:
                        ans = int(1 + inx[final.index(min(final))])
                    except Exception:
                        ans = int(1 + diff.index(min(diff)))
                else:
                    ans = int(1 + rdiff.index(min(rdiff))) if rdiff else 1
            elif 1.0 > refLR(a_image, c_image):
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = refLR(b_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + diff.index(min(diff)))
            elif diss_obj(a_image, b_image):
                diff_ab = diff_finder(a_image, b_image)
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = diff_finder(c_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + min(enumerate(diff), key=lambda x: abs(x[1] - diff_ab))[0])
            elif diss_obj(a_image, c_image):
                diff_ac = diff_finder(a_image, c_image)
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        instance = diff_finder(b_image, PIM.open(problem.figures[i].visualFilename).convert('L'))
                        diff.append(instance)
                ans = int(1 + min(enumerate(diff), key=lambda x: abs(x[1] - diff_ac))[0])
            elif diss_obj(a_image, b_image) and refLR(a_image, b_image) < refTB(a_image, c_image):
                diff = []
                for i in sorted(problem.figures):
                    if i.isnumeric():
                        intermediate = PIM.open(problem.figures[i].visualFilename).convert('L')
                        if diss_obj(c_image, intermediate) and refLR(c_image, intermediate) < refTB(b_image,
                                                                                                        intermediate):
                            instance = refLR(c_image, intermediate)
                            diff.append(instance)
                ans = int(1 + diff.index(min(diff)))
        elif problem.problemType == '3x3':
            fgrs, sltns = begin(problem)
            scrs = []
            if (round(rms(fgrs[1], fgrs[5]), 0) + round(rms(fgrs[0], fgrs[4]), 0) + round(rms(fgrs[3], fgrs[7]),0)) / 3 < 964:  # Used in C & D
                scrs = scr_arr(compare_diag(scrs, fgrs, sltns, problem))
                ans = scrs.index(max(scrs)) + 1
            elif comp_adcf(fgrs) and comp_bcef(fgrs):  # Used in E and Basic C-05
                scrs = scr_arr(top_comp_bttm(scrs, fgrs, sltns))
                ans = scrs.index(max(scrs)) + 1
            elif 'E' in problem.name.split(' ')[2]:  # Used in E
                scrs = img_op_solver(fgrs, sltns)  # Gen & Test
                ans = scrs.index(max(scrs)) + 1
            else:  # Used in C and D
                scrs = gn_tst(smntc_net(fgrs), scrs, fgrs, sltns, problem)
                ans = scrs.index(max(scrs)) + 1
        return ans
