import pandas as pd


def make_folds():
    df = pd.read_csv('tables/stage_2_train.csv')
    mapping_df = pd.read_csv('tables/kaggle_to_nih_id.csv')
    mapping_df['pat_id'] = [int(el.split('_')[0]) for el in mapping_df['NIH_ID']]

    png2id = dict(zip(mapping_df['Kaggle_ID'], mapping_df['pat_id']))

    df['pat_id'] = [png2id.get(el) for el in df['ImageId']]
    print(df.head())
    print(sorted(df['pat_id'].unique()))

    pos, neg = [], []
    for n, group in df.groupby('pat_id'):
        labels = sorted(set(group['EncodedPixels']))
        pat_id = group['pat_id'].tolist()[0]
        if labels == ['-1']:
            neg.append(pat_id)
        else:
            pos.append(pat_id)

    print(len(neg))
    print(len(pos))
    n_fold = 8
    fold_template = list(range(n_fold)) * 100500
    df['fold_id'] = fold_template[:df.shape[0]]

    lst_df = []
    for pat_ids in [pos, neg]:
        id2fold = dict(zip(pat_ids, fold_template))
        tdf = df[df['pat_id'].isin(pat_ids)]
        tdf['fold_id'] = [id2fold.get(el) for el in tdf['pat_id']]
        lst_df.append(tdf)

    df = pd.concat(lst_df)
    print(df.shape)
    print(df.head)

    df.to_csv('tables/folds_v6_st2.csv', index=False)


def check_folds():
    df = pd.read_csv('tables/folds_v6_st2.csv')
    for fold in df['fold_id'].unique():
        train = df[df['fold_id'] != fold]
        valid = df[df['fold_id'] == fold]
        print(train.shape, valid.shape)

        print([el for el in train['pat_id'].tolist() if el in valid['pat_id'].tolist()])

    for pat_id in df['pat_id'].unique():
        tdf = df[df['pat_id'] == pat_id]
        if len(sorted(set(tdf['fold_id']))) > 1:
            print(tdf)


if __name__ == '__main__':
    make_folds()
    check_folds()
