import numpy as np

def expo_nc(t, mk, mi, rounds, sr):
    return np.exp(-(t*(np.square(mk-mi)))/(rounds*sr*2))


def expo_c(t, tc,  t_s, t_e, mk_bc, mk_ac, mi, rounds, sr, p12 = False, time = 0):
    # t: total time, tc: change time, t_s: start time of the round, t_e: end time of the round
    if p12:
        tr = t/(rounds*sr)
        t1 = time
    else:
        tr = t_e-t_s
        t1 = tc-t_s
    delta = (t1*(mi-mk_bc)+(tr-t1)*(mi-mk_ac))/tr
    return np.exp(-(t*(np.square(delta)))/(rounds*sr*2))


def get_probs(config, best_arms, times, chng_time, t, probs, round):
    rc, rs = config.rc, config.rs
    if rc<=rs:
        if t<chng_time:
            # print('1')
            probs['p11'].append(min(1,calc_p11(config, best_arms)))
        elif t>chng_time and probs['p12']==0:
            # print('2')
            probs['p12'] = calc_p12(config, best_arms, times[0], times[1], chng_time)
        else:
            # print('3')
            probs['p13'].append(calc_p13(config, best_arms))
    else:
        if t<chng_time and round<=rs:
            # print('4')
            probs['p21'].append(min(1,calc_p11(config, best_arms)))
        elif t < chng_time and round>rs:
            # print('5')
            probs['p22'].append(1)
        elif round==rc:
            # print('6')
            probs['p23'] = calc_p12(config, best_arms, times[0], times[1], chng_time)
        else:
            # print('7')
            probs['p24'].append(calc_p13(config, best_arms))

def calc_p11(config, best_arm):
    k = config.chng_arm
    m1 = config.mean_dist
    arms = config.arms
    rounds = int(np.log2(arms))
    sr = len(best_arm)
    ret, j = 0, 0
    for i in range(sr):
        if m1[i]<m1[k]:
            ret += expo_nc(config.time, m1[i], m1[k], rounds, sr)
            j+=1
    if k in best_arm:        
        ret +=sr-j-1
    else: ret +=sr-j
    # print(sr-j-1)
    ret *= 2/sr
    return ret

def calc_p12(config, best_arm, ts, te, tc):
    k = config.chng_arm
    m1 = config.mean_dist
    m2 = config.new_mean_dist
    arms = config.arms
    rounds = int(np.log2(arms))
    sr = len(best_arm)
    trc = config.time/(rounds*sr)
    ret = 0
    for i in range(sr):
        ret1 = 0
        tjs = trc*(m2[k]-m1[i])/(m2[k]-m1[k])
        for t in range(int(trc)):
            if t<tjs: ret1 += expo_c(config.time, tc, ts, te, m1[k], m2[k], m1[i], rounds, sr, p12=True, time=t)
            else: ret1 += 1
        ret1 *= 1/trc
        ret += ret1
    ret *= 2/sr
    return ret

def calc_p13(config, best_arm):
    k = config.chng_arm
    m1 = config.mean_dist
    m2 = config.new_mean_dist
    arms = config.arms
    rounds = int(np.log2(arms))
    sr = len(best_arm)
    ret, j = 0, 0
    for i in range(sr):
        if m1[i]<m2[k]:
            ret += expo_nc(config.time, m1[i], m2[k], rounds, sr)
            j+=1
    ret *= 2/sr
    return ret
