from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from readData_yelp import read_yelp
from readData_fungo import read_fungo
from tree import Tree


def evaluate(pred, Y):
    print(f1_score(Y, pred, average='micro'))
    print(f1_score(Y, pred, average='macro'))
    print(f1_score(Y, pred, average='samples'))
    print(classification_report(Y, pred))


def run(X_train, Y_train, X_test, Y_test):
    print('start training')
    model = OneVsRestClassifier(LinearSVC(loss='hinge'), n_jobs=5)
    model.fit(X_train, Y_train)
    pred = model.predict(X_train)
    print('eval training')
    evaluate(pred, Y_train)
    print('eval testing')
    pred = model.predict(X_test)
    evaluate(pred, Y_test)


def rcv1_test():
    from sklearn.datasets import fetch_rcv1
    rcv1 = fetch_rcv1()
    X_train = rcv1.data[:23149]
    Y_train = rcv1.target[:23149]
    X_test = rcv1.data[23149:]
    Y_test = rcv1.target[23149:]
    print(Y_train[:2])
    print(rcv1.target_names[34], rcv1.target_names[59])
    return X_train, Y_train, X_test, Y_test


def yelp_test():
    subtree_name = 'root'
    X_train, X_test, train_ids, test_ids, business_dict, nodes = read_yelp(
        subtree_name, 5, 10)
    print(f'#training={len(train_ids)} #test={len(test_ids)}')
    n_tokens = 256
    print(f'use only first {n_tokens} tokens')
    X_train = [' '.join(i.split()[:n_tokens]) for i in X_train]
    X_test = [' '.join(i.split()[:n_tokens]) for i in X_test]
    print('fit_transform...')
    tf = TfidfVectorizer()
    X_train = tf.fit_transform(X_train)
    X_test = tf.transform(X_test)
    Y_train = [business_dict[bid]['categories'] for bid in train_ids]
    Y_test = [business_dict[bid]['categories'] for bid in test_ids]
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.transform(Y_test)
    return X_train, Y_train, X_test, Y_test


def fungo_test_wrapper(name='cellcycle_FUN'):
    X_train, X_test, train_ids, test_ids, id2doc, nodes = read_fungo(name)
    X_train, X_test = np.array(X_train), np.array(X_test)
    id2doc_train = id2doc
    args = conf()
    # id2doc_train = filter_ancestors(id2doc, nodes)
    tree = Tree(args, train_ids, test_ids, id2doc=id2doc_train, id2doc_a=id2doc, nodes=nodes, rootname='Top')
    mlb = MultiLabelBinarizer(classes=tree.class_idx)
    Y_train = mlb.fit_transform([tree.id2doc_ancestors[docid]['class_idx'] for docid in train_ids])
    Y_test = mlb.transform([tree.id2doc_ancestors[docid]['class_idx'] for docid in test_ids])
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test = yelp_test()
    X_train, Y_train, X_test, Y_test = rcv1_test()
    run(X_train, Y_train, X_test, Y_test)
