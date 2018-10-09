# -*- coding=utf-8 -*-
import pandas as pd
import lightgbm as lgb
from scipy import sparse
import datetime
import numpy as np
import gc
import os
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
import warnings

warnings.filterwarnings("ignore")


def read_data():
    train = pd.read_csv(path + '/data/round1_iflyad_train.txt', sep="\t")
    test = pd.read_csv(path + '/data/round1_iflyad_test_feature.txt', sep="\t")
    return train, test


# 评估函数   查看每个类别的f1值
def timestamp_datatime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def base2(data):
    # ---------------------------------特征删除，全是0-------------------------
    del data['creative_is_js']
    del data['creative_is_voicead']
    del data['app_paid']
    del data['os_name']
    # -------------------------------------time转换---------------------------------------------------
    data['realtime'] = data['time'].apply(timestamp_datatime)

    data['month'] = data['realtime'].apply(lambda x: int(x[5:7])).astype(np.int8)
    data['day'] = data['realtime'].apply(lambda x: int(x[8:10])).astype(np.int8)
    data['day'] = data['day'].apply(lambda x: x + 31 if x < 27 else x)
    data['hour'] = data['realtime'].apply(lambda x: int(x[11:13])).astype(np.int8)
    data['advert_industry_inner_first'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
    data['advert_industry_inner_second'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[1])
    del data['advert_industry_inner']
    del data['realtime']
    data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values
    # 修正os_osv
    tmp = data.groupby('os_osv')['click'].agg({'count'}).reset_index().sort_values('count')
    t = tmp['os_osv'][tmp['count'] == 1].tolist()
    data['os_osv'] = data['os_osv'].apply(lambda x: -1 if x in t else x)
    # ------------------------------------------------------------------------------------------
    del data['os']
    del data['osv']

    # 暂时不处理user_tags
    def delete_null(x):
        return ",".join([s for s in x.split(',') if not s.strip() == ''])

    data['user_tags'] = data['user_tags'].apply(lambda x: delete_null(str(x)))
    # bool类型编码
    columns = ['creative_has_deeplink', 'creative_is_jump', 'creative_is_download']
    for col in columns:
        data[col] = data[col].apply(lambda x: 1 if x == True else 0).astype(np.int8)
    # city特殊 有一个0值，单独编码
    data['city'] = data['city'].apply(lambda x: -1 if x == 0 else x)
    tmp = data[data['city'] == -1]
    data = data[data['city'] != -1]
    le = preprocessing.LabelEncoder()
    data['city'] = le.fit_transform(data['city'])
    data = pd.concat([data, tmp]).reset_index(drop=True)
    # make特征
    # data['make'] = data['make'].apply(lambda x: re.split(r",| |\+|\-|%", str(x).lower())[0])
    # model 字符串分割后，取第一个作为手机的log
    # ------------------------------------实验make--------------------------------------------
    # 引入了噪声
    # data['make'][data['make'].isnull()] = data['model'][data['make'].isnull()]
    # -----------------------------------------------------------------------------------------
    data['model'] = data['model'].apply(lambda x: str(x).lower())
    data['model'] = data['model'].apply(lambda x: re.split(r",| |\+|\-|%", str(x))[0])
    data['model'] = le.fit_transform(data['model']).astype(np.int16)
    # # 类别编码f_channel 含有非字段
    cate_cols = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second',
                 'advert_name', 'campaign_id', 'creative_id', 'creative_tp_dnf',
                 'app_cate_id', 'f_channel', 'app_id', 'inner_slot_id',
                 'city', 'province', 'make', 'nnt', 'model']
    for col in cate_cols:
        data[col] = data[col].fillna(-1)
    data['creative_mul_w_h'] = data['creative_width'] * data['creative_height']
    data['creative_div_w_h'] = data['creative_width'] / (data['creative_height'] + 1)

    # 手动labelencoder
    ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second',
                       'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf',
                       'creative_has_deeplink', 'creative_is_jump', 'creative_is_download']
    # 媒体特征
    media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
    # 上下文特征
    content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'os_osv', 'make', 'model']
    cate_feature = ad_cate_feature + media_cate_feature + content_cate_feature
    for i in cate_feature:
        print(i)
        data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    return data


def get_media_counts(data):
    cols = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
    col = []
    j = 0
    for i in cols:
        col.append(i)
        tmp = data.groupby(col, as_index=False)['click'].agg({'media_' + 'count_' + str(j): 'count'})
        data = pd.merge(data, tmp, on=col, how='left')
        j += 1

    return data


def get_advert_counts(data):
    # ----------------------------------------------------------------------------------------------------
    print("key==advert_industry_inner_second")
    cols = ['advert_industry_inner_first', 'advert_industry_inner_second', 'advert_id', 'advert_name', 'orderid',
            'adid']
    itemcnt = data.groupby(['advert_industry_inner_second'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['advert_industry_inner_second'], how='left')
    for col in ['advert_id', 'advert_name', 'orderid', 'adid']:
        tmp = data.groupby(['advert_industry_inner_second', col], as_index=False)['click'].agg(
            {'advert_industry_inner_second_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['advert_industry_inner_second', col], how='left')
        data[str(col) + '_advert_industry_inner_second_prob'] = data['advert_industry_inner_second_cnt_' + str(col)] / data['item_cnt']
        # del data['advert_industry_inner_second_cnt_' + str(col)]
    del data['item_cnt']

    print("key==advert_id")
    cols = ['advert_id', 'advert_name', 'orderid', 'adid']
    itemcnt = data.groupby(['advert_id'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['advert_id'], how='left')
    for col in ['advert_name', 'orderid', 'adid']:
        tmp = data.groupby(['advert_id', col], as_index=False)['click'].agg(
            {'advert_id_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['advert_id', col], how='left')
        data[str(col) + '_advert_id_prob'] = data['advert_id_cnt_' + str(col)] / data['item_cnt']
        # del data['advert_id_cnt_' + str(col)]
    del data['item_cnt']

    print("key==advert_name")
    cols = ['advert_name', 'orderid', 'adid']
    itemcnt = data.groupby(['advert_name'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['advert_name'], how='left')
    for col in ['orderid', 'adid']:
        tmp = data.groupby(['advert_name', col], as_index=False)['click'].agg(
            {'advert_name_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['advert_name', col], how='left')
        data[str(col) + '_advert_name_prob'] = data['advert_name_cnt_' + str(col)] / data['item_cnt']
        # del data['advert_name_cnt_' + str(col)]
    del data['item_cnt']

    print("key==orderid")
    cols = ['orderid', 'adid']
    itemcnt = data.groupby(['orderid'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['orderid'], how='left')
    for col in ['adid']:
        tmp = data.groupby(['orderid', col], as_index=False)['click'].agg({'orderid_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['orderid', col], how='left')
        data[str(col) + '_orderid_prob'] = data['orderid_cnt_' + str(col)] / data['item_cnt']
        # del data['orderid_cnt_' + str(col)]
    del data['item_cnt']
    # -------------------------------------------------------------------------------------------------------------

    print("key==creative_type")
    cols = ['creative_type', 'campaign_id', 'creative_id']
    itemcnt = data.groupby(['creative_type'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['creative_type'], how='left')
    for col in ['campaign_id', 'creative_id']:
        tmp = data.groupby(['creative_type', col], as_index=False)['click'].agg(
            {'creative_type_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['creative_type', col], how='left')
        data[str(col) + '_creative_type_prob'] = data['creative_type_cnt_' + str(col)] / data['item_cnt']
        # del data['creative_type_cnt_' + str(col)]
    del data['item_cnt']

    print("key==campaign_id")
    cols = ['campaign_id', 'creative_id']
    itemcnt = data.groupby(['campaign_id'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['campaign_id'], how='left')
    for col in ['creative_id']:
        tmp = data.groupby(['campaign_id', col], as_index=False)['click'].agg(
            {'campaign_id_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['campaign_id', col], how='left')
        data[str(col) + '_campaign_id_prob'] = data['campaign_id_cnt_' + str(col)] / data['item_cnt']
        # del data['campaign_id_cnt_' + str(col)]
    del data['item_cnt']

    return data


def get_user_tags_top5(data):
    tmp = data['user_tags'].apply(lambda x: str(x).split(',')).values
    property_dict = {}
    property_list = []
    for i in tmp:
        property_list += i
    for i in property_list:
        if i in property_dict:
            property_dict[i] += 1
        else:
            property_dict[i] = 1
    print('dict finish')

    def top(x):
        propertys = x.split(',')
        cnt = [property_dict[i] for i in propertys]
        res = sorted(zip(propertys, cnt), key=lambda x: x[1], reverse=True)
        top1 = res[0][0]
        top2 = '_'.join([i[0] for i in res[:2]])
        top3 = '_'.join([i[0] for i in res[:3]])
        top4 = '_'.join([i[0] for i in res[:4]])
        top5 = '_'.join([i[0] for i in res[:5]])
        top10 = '_'.join([i[0] for i in res[:10]])
        return (top1, top2, top3, top4, top5, top10)

    data['top'] = data['user_tags'].apply(lambda x: top(str(x)))
    print('top finish')
    data['top1'] = data['top'].apply(lambda x: x[0])
    data['top2'] = data['top'].apply(lambda x: x[1])
    data['top3'] = data['top'].apply(lambda x: x[2])
    data['top4'] = data['top'].apply(lambda x: x[3])
    data['top5'] = data['top'].apply(lambda x: x[4])
    # data['top10'] = data['top'].apply(lambda x: x[5])
    return data


def get_user(data):
    data['property_num'] = data['user_tags'].apply(lambda x: len(str(x).split(',')))
    # 对user_tags内的内容的前五位和第十位

    data['user_tags'] = data['user_tags'].fillna(-1)
    data = get_user_tags_top5(data)
    lb = preprocessing.LabelEncoder()
    top = ['top1', 'top2', 'top3', 'top4', 'top5']
    for col in top:
        data[col] = lb.fit_transform(data[col])

    return data


def get_tfidf_user_tags(data):
    vec = TfidfVectorizer()
    tf_user_tags = vec.fit_transform(data['user_tags'])
    len = data[data['click'] != -1].shape[0]
    tf_train = tf_user_tags[0:len]
    tf_test = tf_user_tags[len:]
    clf = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=24,
        depth=2,
        learning_rate=0.1,
        seed=2018,
        colsample_bytree=0.5,
        subsample=0.8,
        n_jobs=-1,
        n_estimators=100)
    # clf=LogisticRegression()
    clf.fit(tf_train, data.loc[data['click'] != -1, 'click'])
    data['tf_user_tags'] = clf.predict_proba(tf_user_tags)[:, 1]

    return data


def run_before_after_features(data):
    cols = ['adid', 'advert_id', 'advert_industry_inner', 'campaign_id', 'creative_type', 'time', 'day']
    features = []
    for index, row in data.iterrows():
        feature = {}
        feature['instance_id'] = row['instance_id']
        before_adid_cnt = len(
            data[(data['adid'] == row['adid']) & (data['time'] < row['time']) & (data['day'] <= row['day'])])
        before_advert_id_cnt = len(
            data[(data['advert_id'] == row['advert_id']) & (data['time'] < row['time']) & (data['day'] <= row['day'])])
        before_advert_industry_inner_cnt = len(data[(data['advert_industry_inner'] == row['advert_industry_inner']) & (
        data['time'] < row['time']) & (data['day'] <= row['day'])])
        before_campaign_id_inner_cnt = len(data[(data['campaign_id'] == row['campaign_id']) & (
        data['time'] < row['time']) & (data['day'] <= row['day'])])
        before_creative_type_cnt = len(data[(data['creative_type'] == row['creative_type']) & (
        data['time'] < row['time']) & (data['day'] <= row['day'])])

        after_adid_cnt = len(
            data[(data['adid'] == row['adid']) & (data['time'] > row['time']) & (data['day'] <= row['day'])])
        after_advert_id_cnt = len(
            data[(data['advert_id'] == row['advert_id']) & (data['time'] > row['time']) & (data['day'] <= row['day'])])
        after_advert_industry_inner_cnt = len(data[(data['advert_industry_inner'] == row['advert_industry_inner']) & (
        data['time'] > row['time']) & (data['day'] <= row['day'])])
        after_campaign_id_cnt = len(data[(data['campaign_id'] == row['campaign_id']) & (data['time'] > row['time']) & (
        data['day'] <= row['day'])])
        after_creative_type_cnt = len(data[(data['creative_type'] == row['creative_type']) & (
        data['time'] > row['time']) & (data['day'] <= row['day'])])

        feature['before_adid_cnt'] = before_adid_cnt
        feature['before_advert_id_cnt'] = before_advert_id_cnt
        feature['before_advert_industry_inner_cnt'] = before_advert_industry_inner_cnt
        feature['before_campaign_id_inner_cnt'] = before_campaign_id_inner_cnt
        feature['before_creative_type_cnt'] = before_creative_type_cnt

        feature['after_adid_cnt'] = after_adid_cnt
        feature['after_advert_id_cnt'] = after_advert_id_cnt
        feature['after_advert_industry_inner_cnt'] = after_advert_industry_inner_cnt
        feature['after_campaign_id_cnt'] = after_campaign_id_cnt
        feature['after_creative_type_cnt'] = after_creative_type_cnt

        features.append(feature)
    features = pd.DataFrame(features)
    return features


def get_time_feats(data):
    features = pd.DataFrame()
    features = run_before_after_features(data)
    data = pd.merge(data, features, on=['instance_id'], how='left')
    return data


def get_user_before_last(data):
    def run_before_after_features_2(data):
        cols = ['adid', 'advert_id', 'advert_industry_inner', 'campaign_id', 'creative_type', 'time', 'day']
        features = []
        for index, row in data.iterrows():
            feature = {}
            feature['instance_id'] = row['instance_id']
            tmp = data[data['user_id'] == row['user_id']][['instance_id'] + cols]
            before_adid_cnt = len(
                tmp[(tmp['adid'] == row['adid']) & (tmp['time'] < row['time']) & (tmp['day'] <= row['day'])])
            before_advert_id_cnt = len(tmp[(tmp['advert_id'] == row['advert_id']) & (tmp['time'] < row['time']) & (
                tmp['day'] <= row['day'])])
            before_advert_industry_inner_cnt = len(tmp[(tmp['advert_industry_inner'] == row[
                'advert_industry_inner']) & (tmp['time'] < row['time']) & (tmp['day'] <= row['day'])])
            before_campaign_id_inner_cnt = len(tmp[(tmp['campaign_id'] == row['campaign_id']) & (
                tmp['time'] < row['time']) & (tmp['day'] <= row['day'])])
            before_creative_type_cnt = len(tmp[(tmp['creative_type'] == row['creative_type']) & (
                tmp['time'] < row['time']) & (tmp['day'] <= row['day'])])

            after_adid_cnt = len(
                tmp[(tmp['adid'] == row['adid']) & (tmp['time'] > row['time']) & (tmp['day'] <= row['day'])])
            after_advert_id_cnt = len(tmp[(tmp['advert_id'] == row['advert_id']) & (tmp['time'] > row['time']) & (
                tmp['day'] <= row['day'])])
            after_advert_industry_inner_cnt = len(tmp[(tmp['advert_industry_inner'] == row[
                'advert_industry_inner']) & (tmp['time'] > row['time']) & (tmp['day'] <= row['day'])])
            after_campaign_id_cnt = len(tmp[(tmp['campaign_id'] == row['campaign_id']) & (
                tmp['time'] > row['time']) & (tmp['day'] <= row['day'])])
            after_creative_type_cnt = len(tmp[(tmp['creative_type'] == row['creative_type']) & (
                tmp['time'] > row['time']) & (tmp['day'] <= row['day'])])

            feature['userid_before_adid_cnt'] = before_adid_cnt
            feature['userid_before_advert_id_cnt'] = before_advert_id_cnt
            feature['userid_before_advert_industry_inner_cnt'] = before_advert_industry_inner_cnt
            feature['userid_before_campaign_id_inner_cnt'] = before_campaign_id_inner_cnt
            feature['userid_before_creative_type_cnt'] = before_creative_type_cnt

            feature['userid_after_adid_cnt'] = after_adid_cnt
            feature['userid_after_advert_id_cnt'] = after_advert_id_cnt
            feature['userid_after_advert_industry_inner_cnt'] = after_advert_industry_inner_cnt
            feature['userid_after_campaign_id_cnt'] = after_campaign_id_cnt
            feature['userid_after_creative_type_cnt'] = after_creative_type_cnt

            features.append(feature)
        features = pd.DataFrame(features)
        return features

    features = pd.DataFrame()
    features = run_before_after_features_2(data)
    data = pd.merge(data, features, on=['instance_id'], how='left')
    return data


def get_leak_feature(data):
    return data


def get_cvr(data):
    # 历史转化率
    for feat_1 in ['advert_id', 'advert_industry_inner_second', 'advert_name',
                   'campaign_id', 'creative_height',
                   'creative_tp_dnf', 'creative_width', 'province', 'f_channel', ]:
        gc.collect()
        res = pd.DataFrame()
        temp = data[[feat_1, 'hour', 'click']][data['click'] != -1]

        for h in range(0, 24):
            count = temp.groupby([feat_1]).apply(
                lambda x: x['click'][(x['hour'] == h)].count()).reset_index(name=feat_1 + '_all')
            count.fillna(value=0, inplace=True)
            # count[feat_1 + '_rate'] = round(count[feat_1 + '_1'] / (10000+count[feat_1 + '_all']),5)
            count['hour'] = h
            # count.drop([feat_1 + '_all',feat_1+'_1'], axis=1, inplace=True)
            count.fillna(value=0, inplace=True)
            res = res.append(count, ignore_index=True)
        print(feat_1, ' over')
        data = pd.merge(data, res, how='left', on=[feat_1, 'hour'])
        data[feat_1 + '_all'] = data[feat_1 + '_all'].fillna(0)
        data[feat_1 + '_all'][data[feat_1 + '_all'] <= 100] = 0

    return data


def get_statis_period_feats(data):
    cols = ['adid', 'advert_id', 'advert_industry_inner', 'creative_width', 'creative_height',
            # 'inner_slot_id','app_id','f_channel','app_cate_id',
            # 'city','province','alians_model','make','os'
            ]
    for col in cols:
        gc.collect()
        tmp = data.groupby(by=[col, 'period']).apply(lambda x: x['click'][x['click'] == 1].sum()).reset_index(
            name='cnt_byday_' + col)
        data = pd.merge(data, tmp, on=[col, 'period'], how='left')
        data['cnt_byday_' + col] = data['cnt_byday_' + col].astype(np.int16)
        data['cnt_byday_' + col] = data['cnt_byday_' + col].fillna(0)
        print(col + 'statis is over')

    # cols = ['adid', 'advert_id', 'advert_industry_inner_0', 'advert_industry_inner', 'creative_width',
    #         'creative_height',
    #         'inner_slot_id', 'app_id', 'f_channel', 'app_cate_id',
    #         'city', 'province', 'alians_model', 'make', 'os']
    # for col in cols:
    #     # 统计特征 mean
    #     tmp = data.groupby(by=[col],as_index=False)['cnt_byday_'+col].agg({col+'_mean':'mean',col+'_std':'std',
    #                     col + '_var': 'var',col+'_max':'max',col+'_min':'min',col+'_std':'std'
    #                                                                        })
    #     data=pd.merge(data,tmp,on=[col],how='left')

    return data


def construct_user_id(data):
    # 构造相似人群
    lb = preprocessing.LabelEncoder()
    data['user_id'] = lb.fit_transform(data['top'])

    # user_id 偏好
    for feat_1 in ['advert_id', 'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_height',
                   'creative_tp_dnf', 'creative_width', 'province', 'f_channel', 'city', 'osv']:
        gc.collect()
        res = pd.DataFrame()
        res = data.groupby(['user_id', feat_1], as_index=False)['click'].agg({'user_cnt_' + feat_1: 'count'})
        data = pd.merge(data, res, how='left', on=['user_id', feat_1])
        data['user_cnt_' + feat_1] = data['user_cnt_' + feat_1].fillna(0).astype(np.int32)
        print('user_id' + ' ' + feat_1 + ' is over')

    return data


def single_feat_count(data):
    cols = ['province', 'city', 'inner_slot_id', 'app_id', 'f_channel',
            'app_cate_id', 'creative_width', 'creative_height', 'creative_tp_dnf', 'creative_type',
            # 'creative_id','campaign_id','advert_name','advert_industry_inner','orderid',
            # 'advert_id','adid',
            ]
    for col in cols:
        print('single ' + col)
        tmp = data.groupby(col, as_index=False)['click'].agg({'single_' + col + '_cnt': 'count'})
        data = pd.merge(data, tmp, on=[col], how='left')
        data['single_' + col + '_cnt'] = data['single_' + col + '_cnt'].fillna(0).astype(np.int32)
    return data


def get_industry_inner(data):
    # advert_industry_inner特征
    # 按照官方说的, 这个特征有两个部分, 一般这种特征的话大家常见的操作有三个，一个是split展开，第二个是统计后半部分的出现次数, count, 第三个就是统计后半部分出现次数占整个类的比例。
    tmp = data.groupby('advert_industry_inner_first')['click'].agg({'count'}).reset_index().rename(
        columns={'count': 'first_cnt'})
    data = pd.merge(data, tmp, on=['advert_industry_inner_first'], how='left')
    tmp = data.groupby('advert_industry_inner_second')['click'].agg({'count'}).reset_index().rename(
        columns={'count': 'second_cnt'})
    data = pd.merge(data, tmp, on=['advert_industry_inner_second'], how='left')
    data['first_second_prob'] = data['first_cnt'] / data['second_cnt']
    return data


def convert2sparse(data):
    # 初始特征
    # 广告特征

    ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first', 'advert_industry_inner_second',
                       'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf',
                       'creative_has_deeplink', 'creative_is_jump', 'creative_is_download']
    # 媒体特征
    media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
    # 上下文特征
    content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'os_osv', 'make', 'model']
    cate_feature = ad_cate_feature + media_cate_feature + content_cate_feature

    # 不能训练特征
    not_use = ['instance_id', 'click', 'time', 'user_tags',]
    # 数值特征
    num_feature = [i for i in data if i not in not_use and i not in cate_feature]

    print('num_feature数值特征', num_feature)
    print('num_feature特征个数', len(num_feature))

    predict = data[data.click == -1]
    predict_result = predict[['instance_id']]
    predict_result['predicted_score'] = 0
    predict_x = predict.drop('click', axis=1)

    train_x = data[data.click != -1]
    train_y = data[data.click != -1].click.values
    if os.path.exists(path + '/feature/base_train_csr.npz') and False:
        print('load_csr---------')
        base_train_csr = sparse.load_npz(path + '/feature/base_train_csr.npz').tocsr().astype('bool')
        base_predict_csr = sparse.load_npz(path + '/feature/base_predict_csr.npz').tocsr().astype('bool')
    else:
        base_train_csr = sparse.csr_matrix((len(train_x), 0))
        base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

        enc = preprocessing.OneHotEncoder()
        for feature in cate_feature:
            print(feature)
            enc.fit(data[feature].values.reshape(-1, 1))
            base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))),'csr', 'bool')
            base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),'csr', 'bool')
        print('one-hot prepared !')
        cv = CountVectorizer(min_df=20)
        for feature in ['user_tags']:
            data[feature] = data[feature].astype(str)
            cv.fit(data[feature])
            base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
            base_predict_csr=sparse.hstack((base_predict_csr,cv.transform(predict_x[feature].astype(str))), 'csr','bool')
        print('cv prepared !')
        sparse.save_npz(path + '/feature/base_train_csr.npz', base_train_csr)
        sparse.save_npz(path + '/feature/base_predict_csr.npz', base_predict_csr)

    train_csr = sparse.hstack((sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype('float32')
    predict_csr = sparse.hstack((sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
    print('train_csr的shape', train_csr.shape)

    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(train_csr, train_y)
    train_csr = feature_select.transform(train_csr)
    predict_csr = feature_select.transform(predict_csr)
    print('feature select')
    print(train_csr.shape)

    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=48, reg_alpha=3, reg_lambda=5,
        max_depth=-1, n_estimators=5000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, random_state=22, n_jobs=-1,
        min_child_weight=5, min_child_samples=10,
        # max_bin=425, subsample_for_bin=50000,

    )
    n = 5
    skf = StratifiedKFold(n_splits=n, random_state=10, shuffle=True)
    best_score = []
    for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
        # print(index,'index-------------------')
        lgb_model.fit(train_csr[train_index], train_y[train_index],
                      eval_set=[(train_csr[train_index], train_y[train_index]),
                                (train_csr[test_index], train_y[test_index])], early_stopping_rounds=100)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
        print('test mean:', test_pred.mean())
        predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

        # print('----------特征重要性-------------')
        # predictors = [i for i in train_csr.columns]
        # 仅仅获取非稀疏矩阵重要性
        feat_imp = pd.Series(lgb_model.feature_importances_[0:len(num_feature)], num_feature).sort_values(ascending=False)
        # print(feat_imp)
        feat_imp.to_csv(path + '/result/feat_importance_%s.csv'%index)

    print(np.mean(best_score))
    predict_result['predicted_score'] = predict_result['predicted_score'] / n
    mean = predict_result['predicted_score'].mean()
    print('mean:', mean)
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    predict_result[['instance_id', 'predicted_score']].to_csv(path + "/result/lgb_baseline_%s.csv" % now, index=False)

    stop_time = time.time() - start_time
    print("训练时间", stop_time)


def cross_feats(data):
    a = ['province',
         # 'city',
         # 'nnt','os_osv'
         ]
    b = ['model']
    for i in a:
        for j in b:
            data[i + '_' + j + '_cross'] = round((data[i]+0.5)/ (0.7 + data[j]), 5)
    return data


def get_context_cnt(data):
    print("key==carrier")
    cols = ['carrier', 'devtype', 'nnt', 'province', 'os_osv', 'city', 'make', 'model']
    itemcnt = data.groupby(['carrier'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['carrier'], how='left')
    for col in ['devtype', 'nnt', 'province', 'os_osv', 'city', 'make', 'model']:
        tmp = data.groupby(['carrier', col], as_index=False)['click'].agg(
            {'carrier_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['carrier', col], how='left')
        data[str(col) + '_carrier_prob'] = data['carrier_cnt_' + str(col)] / data['item_cnt']
        # del data['carrier_cnt_' + str(col)]
    del data['item_cnt']

    print("key==devtype")
    cols = ['nnt', 'province', 'os_osv', 'city', 'make', 'model']
    itemcnt = data.groupby(['devtype'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['devtype'], how='left')
    for col in ['nnt', 'province', 'os_osv', 'city', 'make', 'model']:
        tmp = data.groupby(['devtype', col], as_index=False)['click'].agg(
            {'devtype_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['devtype', col], how='left')
        data[str(col) + '_devtype_prob'] = data['devtype_cnt_' + str(col)] / data['item_cnt']
        # del data['devtype_cnt_' + str(col)]
    del data['item_cnt']

    print("key==nnt")
    cols = ['province', 'os_osv', 'city', 'make', 'model']
    itemcnt = data.groupby(['nnt'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['nnt'], how='left')
    for col in ['province', 'os_osv', 'city', 'make', 'model']:
        tmp = data.groupby(['nnt', col], as_index=False)['click'].agg(
            {'nnt_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['nnt', col], how='left')
        data[str(col) + '_nnt_prob'] = data['nnt_cnt_' + str(col)] / data['item_cnt']
        # del data['nnt_cnt_' + str(col)]
    del data['item_cnt']

    print("key==province")
    cols = ['os_osv', 'city', 'make', 'model']
    itemcnt = data.groupby(['province'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['province'], how='left')
    for col in ['os_osv', 'city', 'make', 'model']:
        tmp = data.groupby(['province', col], as_index=False)['click'].agg({'province_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['province', col], how='left')
        data[str(col) + '_province_prob'] = data['province_cnt_' + str(col)] / data['item_cnt']
        # del data['province_cnt_' + str(col)]
    del data['item_cnt']
    # -------------------------------------------------------------------------------------------------------------

    print("key==os_osv")
    cols = ['city', 'make', 'model']
    itemcnt = data.groupby(['os_osv'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['os_osv'], how='left')
    for col in ['city', 'make', 'model']:
        tmp = data.groupby(['os_osv', col], as_index=False)['click'].agg(
            {'os_osv_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['os_osv', col], how='left')
        data[str(col) + '_os_osv_prob'] = data['os_osv_cnt_' + str(col)] / data['item_cnt']
        # del data['os_osv_cnt_' + str(col)]
    del data['item_cnt']

    print("key==city")
    cols = ['make', 'model']
    itemcnt = data.groupby(['city'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['city'], how='left')
    for col in ['make', 'model']:
        tmp = data.groupby(['city', col], as_index=False)['click'].agg(
            {'city_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['city', col], how='left')
        data[str(col) + '_city_prob'] = data['city_cnt_' + str(col)] / data['item_cnt']
        # del data['city_cnt_' + str(col)]
    del data['item_cnt']
    return data


def get_media_counts2(data):
    print("key==app_cate_id")
    cols = ['f_channel', 'app_id', 'inner_slot_id']
    itemcnt = data.groupby(['app_cate_id'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['app_cate_id'], how='left')
    for col in ['f_channel', 'app_id', 'inner_slot_id']:
        tmp = data.groupby(['app_cate_id', col], as_index=False)['click'].agg(
            {'app_cate_id_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['app_cate_id', col], how='left')
        data[str(col) + '_app_cate_id_prob'] = data['app_cate_id_cnt_' + str(col)] / data['item_cnt']
        # del data['app_cate_id_cnt_' + str(col)]
    del data['item_cnt']

    print("key==f_channel")
    cols = ['app_id', 'inner_slot_id']
    itemcnt = data.groupby(['f_channel'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['f_channel'], how='left')
    for col in ['app_id', 'inner_slot_id']:
        tmp = data.groupby(['f_channel', col], as_index=False)['click'].agg(
            {'f_channel_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['f_channel', col], how='left')
        data[str(col) + '_f_channel_prob'] = data['f_channel_cnt_' + str(col)] / data['item_cnt']
        # del data['f_channel_cnt_' + str(col)]
    del data['item_cnt']

    print("key==app_id")
    cols = ['inner_slot_id']
    itemcnt = data.groupby(['app_id'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['app_id'], how='left')
    for col in ['inner_slot_id']:
        tmp = data.groupby(['app_id', col], as_index=False)['click'].agg(
            {'app_id_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['app_id', col], how='left')
        data[str(col) + '_app_id_prob'] = data['app_id_cnt_' + str(col)] / data['item_cnt']
        # del data['app_id_cnt_' + str(col)]
    del data['item_cnt']

    return data


def ad_context_cnt(data):
    # a=['advert_id','advert_industry_inner']
    # b=['city','province','devtype','make','model']

    print("key==advert_id")
    cols = ['city', 'province', 'devtype', 'make', 'model']
    itemcnt = data.groupby(['advert_id'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['advert_id'], how='left')
    for col in ['city', 'province', 'devtype', 'make', 'model']:
        tmp = data.groupby(['advert_id', col], as_index=False)['click'].agg(
            {'advert_id_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['advert_id', col], how='left')
        data[str(col) + '_advert_id_prob'] = data['advert_id_cnt_' + str(col)] / data['item_cnt']
        del data['advert_id_cnt_' + str(col)]
    del data['item_cnt']

    print("key==advert_industry_inner_second")
    cols = ['city', 'province', 'devtype', 'make', 'model']
    itemcnt = data.groupby(['advert_industry_inner_second'], as_index=False)['click'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['advert_industry_inner_second'], how='left')
    for col in ['city', 'province', 'devtype', 'make', 'model']:
        tmp = data.groupby(['advert_industry_inner_second', col], as_index=False)['click'].agg(
            {'advert_industry_inner_second_cnt_' + str(col): 'count'})
        data = pd.merge(data, tmp, on=['advert_industry_inner_second', col], how='left')
        data[str(col) + '_advert_industry_inner_second_prob'] = data['advert_industry_inner_second_cnt_' + str(col)] / \
                                                                data['item_cnt']
        del data['advert_industry_inner_second_cnt_' + str(col)]
    del data['item_cnt']
    return data


if __name__ == '__main__':
    # 全局参数
    start_time = time.time()
    path = 'E:/studytools/pycharmProject/xunfei'
    # path = 'C:/dapao/pyCharmProject/xunfei'
    train, test = read_data()
    data = pd.concat([train, test]).reset_index(drop=True)
    data['click'] = data['click'].fillna(-1)
    # ----------------------逻辑处理01---------------------------------
    data = base2(data)
    # -----------------------------------------------------------------
    print('base2执行完之后的特征数', data.shape)
    data = get_media_counts(data)
    print('get_media_counts执行完之后的特征数', data.shape)
    data = get_industry_inner(data)
    print('get_industry_inner执行完之后的特征数', data.shape)
    data = get_advert_counts(data)
    # 统计量特征
    data = get_cvr(data)
    data = get_context_cnt(data)
    data = get_media_counts2(data)
    # ------------------------------待验证特征------------------------------------------------------------------------
    # data=pd.read_csv(path+"/data/data_ef08.csv")
    data = cross_feats(data)
    # ----------------------------------------onehot并转化成稀疏矩阵-+5折训练---------------------------------------------
    data = convert2sparse(data)
