import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
import os
from multiprocessing import Pool
import itertools
from tqdm import trange
import math
import matplotlib.pyplot as plt
import yaml

def get_gt(num_pts:int, num_lbls_per_pt:int, corpus_lbl_prior:np.ndarray) -> sp.csr_matrix:
    """

    Args:
        num_pts (int): Number of documents in sparse matric
        num_lbls_per_pt (int): For each documents these many labels will be sampled
        corpus_lbl_prior (np.ndarray): Prior distribution of label occurances: Ensure it is a distribution, things sum up to 1

    Returns:
        sp.csr_matrix: relevance matrix denoting which docs are associated to which labels
    """

    # nnz = 0
    num_labels = corpus_lbl_prior.shape[0]
    sampling_result = np.random.multinomial(num_lbls_per_pt, corpus_lbl_prior, size=num_pts)
    rows, cols = np.where(sampling_result > 0)
    nnz = len(rows)
    data = np.ones(nnz)
        
    print(f"GT construction done! NNZ: {nnz}")
    X_Y = sp.csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(num_pts, num_labels))
    X_Y.data.fill(1)
    return X_Y


def generate_doc_fts(X_Y: sp.csr_matrix , lbl_class_dist: np.ndarray, lbl_class_token_mapping: np.ndarray, doc_class_dist: np.ndarray, doc_class_token_mapping: np.ndarray) -> np.ndarray:
    """
    Generate document features following LDA like setting

    Args:
        X_Y (sp.csr_matrix): Relevance matrix between documents and labels
        lbl_class_dist (np.ndarray): Each label is associated with `num_tokens_per_topic` number of tokens, this provides the distribution over that
        lbl_class_token_mapping (np.ndarray): Used to map the index sampled from the above distribution to a token index
        doc_class_dist (np.ndarray): Each lbl is alternatively associated with `num_tokens_per_topic` number of labels, this provides the distribution over that and will be used in heterogenous case
        doc_class_token_mapping (np.ndarray): Used to map the index sampled from the above distribution to a token index
        
    Returns:
        np.ndarray: Returns np.int64 2D array containing indices of tokens for the doc
    """
    
    num_data_pts, num_labels = X_Y.shape
    data_pt_texts = []
    for data_pt_index in trange(num_data_pts):
        # Assuming uniform sampling of labels => For high `alpha` in dirichlet this will happen
        data_pt_text = []
        lbl_indices = X_Y[data_pt_index].indices
        num_gt_lbls = len(lbl_indices)
        for position in range(doc_text_seq_len):
            label_index = lbl_indices[np.random.randint(0, num_gt_lbls)]

            homogenity_indicator = np.random.binomial(1, homogenity_factor)
            if homogenity_indicator == 1:
                # Sample from label -> vocab dist
                token_index = lbl_class_token_mapping[label_index][get_class_index_from_multinomial_dist(lbl_class_dist[label_index])]
            else:
                # Sample from doc -> vocab 
                token_index = doc_class_token_mapping[label_index][get_class_index_from_multinomial_dist(doc_class_dist[label_index])]
            data_pt_text.append(token_index)
        data_pt_texts.append(data_pt_text)
    data_pt_texts = np.array(data_pt_texts, dtype=np.int64)
    return data_pt_texts
#           

def generate_lbl_fts(lbl_class_dist: np.ndarray, class_token_mapping: np.ndarray) -> np.ndarray:
    """
    Generates label features given the distribution of associated tokens with each label

    Args:
        lbl_class_dist (np.ndarray): Each label is associated with `num_tokens_per_topic` number of labels, this provides the distribution over that
        class_token_mapping (np.ndarray): Used to map the index sampled from the above distribution to a token index

    Returns:
        np.ndarray: Returns np.int64 2D array containing indices of tokens for the label
    """
    label_texts = []
    for label_index in trange(num_labels):
        label_text = []
        multinomial_dist_parms = lbl_class_dist[label_index]
        for position in range(label_text_seq_len):
            token_index = class_token_mapping[label_index][get_class_index_from_multinomial_dist(multinomial_dist_parms)]
            label_text.append(token_index)
        label_texts.append(label_text)
    label_texts = np.array(label_texts, dtype=np.int64)
    return label_texts

def partition_X_Y(X_Y: sp.csr_matrix, num_processes: int) -> List[sp.csr_matrix]:
    """
    Paritions the X_Y matrix for parallel feature generation (from generate_doc_fts)

    Args:
        X_Y (sp.csr_matrix): original X_Y
        num_processes (int): number of partitions of X_Y 

    Returns:
        List[sp.csr_matrix]: Paritioned components of X_Y
    """
    X_Y_list = []
    num_data_pts = X_Y.shape[0]
    start = 0
    partition_sz = num_data_pts // num_processes
    for partition_index in range(num_processes):
        end = start + partition_sz
        if partition_index == num_processes - 1:
            end = num_data_pts
        X_Y_list.append(X_Y[start: end])
        start = end
    print("Paritioning Done!")
    return X_Y_list

def get_dirichlet_dist(num_classes: int, alpha: float) -> np.ndarray:
    """
    Generates a dirichlet distribution over `num_classes` with given `alpha` as parameter

    Args:
        num_classes (int): Number of classes
        alphas (float): Dirichlet Distribution parameter

    Returns:
        np.ndarray: 1D array containing alphas for Dirichlet distribution
    """
    return np.ones(num_classes) * alpha

def get_lbl_token_dirichlet_dist(num_lbls: int, num_tokens_per_lbl: int, lbl_token_alpha: float, word_pool_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        num_lbls (int): Number of labels
        num_tokens_per_lbl (int): Number of tokens associated with each label
        lbl_token_alpha (float): Dirichlet distribution parameter for the label
        word_pool_size (int): Vocab size

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            * First element is the label class alpha parameter of the dirichlet distribution
            * Second element is the class token mapping
    """
    label_token_alphas_dirichlet = np.ones((num_labels, num_tokens_per_label)) * lbl_token_alpha
    class_token_mapping = np.zeros((num_labels, num_tokens_per_label), dtype=np.int64)
    for label_index in range(num_labels):
        class_token_mapping[label_index] = np.random.randint(0, word_pool_size, num_tokens_per_label)

    return label_token_alphas_dirichlet, class_token_mapping

def get_power_law(k: float, n:int) -> np.ndarray:
    """

    Args:
        k (float): Exponent parameter
        n (int): Number of entities

    Returns:
        np.ndarray: Power law distribution over `n` entities with exponent parameter k
    """
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = math.pow((i + 1), -k)
    dist = dist / np.sum(dist)
    return dist

def get_multinomial_dist_from_dirichlet(alphas: np.ndarray) -> np.ndarray:
    """
    Samples a multinomial distribution from a dirichlet distiribution

    Args:
        alphas (np.ndarray): Dirichlet distribution parameters

    Returns:
        np.ndarray: Multinomial distribution
    """
    multinomial_dist_params = np.zeros(alphas.shape)
    if len(multinomial_dist_params.shape) == 1:
        multinomial_dist_params = np.random.dirichlet(alphas)
    else:
        for i in trange(alphas.shape[0]):
            multinomial_dist_params[i] = np.random.dirichlet(alphas[i])
    return multinomial_dist_params

def get_class_index_from_multinomial_dist(probs: np.ndarray) -> int:
    """
    Samples an element from a multinomial distribution

    Args:
        probs (np.ndarray): Multinomial distribution parameters

    Returns:
        int: Class index sampled from mulinomial dist
    """
    class_index = np.where(np.random.multinomial(1, probs) == 1)[0][0]
    return class_index

def calc_overlap_count(str1: str, str2: str) -> int:
    """
    Calculates how many tokens in str1 are also present in str2 (if one token in str1 matches with 2 of str2 count returned is 2)

    Args:
        str1 (str):
        str2 (str): 

    Returns:
        int: overlap count
    """
    ctr = 0
    for el1 in str1:
        for el2 in str2:
            if el1 == el2:
                ctr += 1
    return ctr

def get_dropped_gt(X_Y:sp.csr_matrix, lbl_dropping_probs: np.ndarray) -> sp.csr_matrix:
    """
    Drops some (doc, lbl) pairs from existing X_Y to generate a X_Y reflective of missing labels

    Args:
        X_Y (sp.csr_matrix): original X_Y
        lbl_dropping_probs (np.ndarray): probability of dropping a label

    Returns:
        sp.csr_matrix: X_Y generated after dropping some pairs
    """
    return

with open('params.yaml','r') as f:
    params_dict = yaml.safe_load(f)

num_topics = params_dict['num_topics']
num_trn_data_pts = params_dict['num_trn_data_pts']
num_tst_data_pts = params_dict['num_tst_data_pts']
num_labels = num_topics
word_pool_size = params_dict['word_pool_size']

homogenity_factor = params_dict['homogenity_factor'] # Will be used for the bernoulli distribution

label_token_alpha_dirichlet = params_dict['label_token_alpha_dirichlet']
# Alphas for topic label dirichlet are set as per a power law a * (x^-k)
lbl_prior_k = params_dict['lbl_prior_k']

topic_sampling_alpha_dirichlet = params_dict['topic_sampling_alpha_dirichlet']
label_text_seq_len = params_dict['label_text_seq_len']
doc_text_seq_len = params_dict['doc_text_seq_len']
labels_sampled_per_data_pt = params_dict['labels_sampled_per_data_pt']
num_tokens_per_label = params_dict['num_tokens_per_label']
synthetic_dataset_dir = params_dict['datasets_dir']
dataset_name = f'num_top_{num_topics}_num_lbls_{num_labels}_vocab_{word_pool_size}_homof_{homogenity_factor}_lbl_tok_alpha_d_{label_token_alpha_dirichlet}_top_lbl_k_{lbl_prior_k}_top_sampling_alpha_d{topic_sampling_alpha_dirichlet}_lbl_seq_len_{label_text_seq_len}_doc_seq_len_{doc_text_seq_len}_lbls_sampled_per_pt_{labels_sampled_per_data_pt}'
dataset_dir = f'{synthetic_dataset_dir}/{dataset_name}'
num_processes = 10

def get_attention_masks() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns the attention mask for trn_doc, tst_doc, lbls
    """
    max_seq_len = max(doc_text_seq_len, label_text_seq_len)
    trn_doc_attention_mask = np.zeros((num_trn_data_pts, max_seq_len)).astype(np.int64)
    tst_doc_attention_mask = np.zeros((num_tst_data_pts, max_seq_len)).astype(np.int64)
    lbl_attention_mask = np.zeros((num_labels, max_seq_len)).astype(np.int64)
    
    trn_doc_attention_mask[:, :doc_text_seq_len] = 1
    tst_doc_attention_mask[:, :doc_text_seq_len] = 1
    lbl_attention_mask[:, :label_text_seq_len] = 1
    return trn_doc_attention_mask, tst_doc_attention_mask, lbl_attention_mask

def save_numpy_arrs(np_arrs: List[np.ndarray], fnames: List[str]):
    """
    Saves all numpy arrs with specified file names

    Args:
        np_arrs (List[np.ndarray]): List of numpy arrays to save
        fnames (List[str]): List of fnames
    """
    os.makedirs(f'{dataset_dir}/artifacts', exist_ok=True)
    for np_arr, fname in zip(np_arrs, fnames):
        np.save(f'{dataset_dir}/artifacts/{fname}.npy', np_arr)
    print(f"Saved all artifacts at: f{dataset_dir}/artifacts")

def save_numpy_arrs_as_memmap(np_arrs: List[np.ndarray], fnames: List[str]) -> None:
    seq_len = max(doc_text_seq_len, label_text_seq_len)
    tok_dir = f'{dataset_dir}/bert-base-uncased-{seq_len}'
    os.makedirs(tok_dir, exist_ok = True)
    for np_arr, fname in zip(np_arrs, fnames):
        mmap = np.memmap(f'{tok_dir}/{fname}', shape=np_arr.shape, dtype=np.int64, mode='w+')
        mmap[:] = np_arr[:]
        assert (mmap - np_arr).sum() == 0


def calc_overlap_stats(X_Y:sp.csr_matrix, data_pt_texts:np.ndarray, lbl_texts:np.ndarray) -> int:
    """

    Args:
        data_pt_texts (np.ndarray): Text of data pts
        lbl_text (np.ndarray): Text of labels

    Returns:
        int: Avg. overlap between data points and labels
    """
    overlap_ctr = 0
    num_pairs = 0
    
    for i in range(1000):
        data_pt_index = np.random.randint(0, num_trn_data_pts)
        lbl_indices = X_Y[i].indices
        for lbl_index in lbl_indices:
            overlap_ctr += calc_overlap_count(data_pt_texts[i], lbl_texts[lbl_index])
            num_pairs += 1
    return overlap_ctr / num_pairs

    

def print_stats(trn_X_Y: sp.csr_matrix, tst_X_Y: sp.csr_matrix, trn_data_pt_texts: np.ndarray, tst_data_pt_texts: np.ndarray, label_texts: np.ndarray) -> None:
    """
    Prints out the stats of generated corpus

    Args:
        trn_X_Y (sp.csr_matrix): Train GT matrix
        tst_X_Y (sp.csr_matrix): Test GT matrix
        trn_data_pt_texts (np.ndarray): Token indices of training data
        tst_data_pt_texts (np.ndarray): Token indices of test data
        label_texts (np.ndarray): Token indices of label text
    """
    print("Calculating stats")
    
    trn_ppl = np.array(trn_X_Y.sum(axis = 0)).flatten()
    tst_ppl = np.array(tst_X_Y.sum(axis = 0)).flatten()
    
    trn_lpp = np.array(trn_X_Y.sum(axis = 1)).flatten()
    tst_lpp = np.array(tst_X_Y.sum(axis = 1)).flatten()
    
    print(f'Avg. Trn PPL: {trn_ppl.mean():2.2f} | Min Trn PPL: {trn_ppl.min():2.2f} | Max Trn PPL: {trn_ppl.max():2.2f} | Avg. Trn LPP: {trn_lpp.mean():2.2f}')
    print(f'Avg. Tst PPL: {tst_ppl.mean():2.2f} | Min Tst PPL: {tst_ppl.min():2.2f} | Max Tst PPL: {tst_ppl.max():2.2f} | Avg. Tst LPP: {tst_lpp.mean():2.2f}')
    
    sorted_trn_ppl = np.sort(trn_ppl)[::-1]
    sorted_tst_ppl = np.sort(tst_ppl)[::-1]
    plt.plot(sorted_trn_ppl)
    plt.savefig('./trn_ppl')
    
    trn_overlap = calc_overlap_stats(trn_X_Y, trn_data_pt_texts, label_texts)
    tst_overlap = calc_overlap_stats(tst_X_Y, tst_data_pt_texts, label_texts)
    print(f"Overlap stats: Trn overlap: {trn_overlap:2.2f} Tst overlap: {tst_overlap:2.2f}")


def generate_data():
    corpus_lbl_prior = get_power_law(lbl_prior_k, num_labels)
    topic_drichlet_dist = get_dirichlet_dist(num_topics, topic_sampling_alpha_dirichlet)
    
    label_token_alphas_dirichlet, label_class_token_mapping = get_lbl_token_dirichlet_dist(num_labels, num_tokens_per_label, label_token_alpha_dirichlet, word_pool_size)
    label_token_multinomial_probs = get_multinomial_dist_from_dirichlet(label_token_alphas_dirichlet)
    
    doc_token_alphas_dirichlet, doc_class_token_mapping = label_token_alphas_dirichlet, label_class_token_mapping
    doc_token_multinomial_probs = label_token_multinomial_probs
    
    is_heterogenous = False
    if homogenity_factor < 1:
        is_heterogenous = True
        doc_token_alphas_dirichlet, doc_class_token_mapping = get_lbl_token_dirichlet_dist(num_labels, num_tokens_per_label, label_token_alpha_dirichlet, word_pool_size)
        doc_token_multinomial_probs = get_multinomial_dist_from_dirichlet(doc_token_alphas_dirichlet)
    
    save_numpy_arrs(
        [
            corpus_lbl_prior, topic_drichlet_dist, label_token_alphas_dirichlet, label_class_token_mapping, 
            label_token_multinomial_probs, doc_token_alphas_dirichlet, doc_class_token_mapping, doc_token_multinomial_probs
        ],
        [
            'corpus_lbl_prior', 'topic_drichlet_dist', 'label_token_alphas_dirichlet', 'label_class_token_mapping',
            'label_token_multinomial_probs', 'doc_token_alphas_dirichlet', 'doc_class_token_mapping', 'doc_token_multinomial_probs'
        ]
    )
    
    label_texts = generate_lbl_fts(label_token_multinomial_probs, label_class_token_mapping)
    trn_X_Y = get_gt(num_trn_data_pts, labels_sampled_per_data_pt, corpus_lbl_prior)
    tst_X_Y = get_gt(num_tst_data_pts, labels_sampled_per_data_pt, corpus_lbl_prior)
    
    p_trn_X_Y_list = partition_X_Y(trn_X_Y, num_processes)
    p_tst_X_Y_list = partition_X_Y(tst_X_Y, num_processes)
    
    trn_args_list = []
    tst_args_list = []
    for proc_indx in range(num_processes):
        trn_proc_args = (p_trn_X_Y_list[proc_indx], label_token_multinomial_probs, label_class_token_mapping, doc_token_multinomial_probs, doc_class_token_mapping)
        tst_proc_args = (p_tst_X_Y_list[proc_indx], label_token_multinomial_probs, label_class_token_mapping, doc_token_multinomial_probs, doc_class_token_mapping)
        
        trn_args_list.append(trn_proc_args)
        tst_args_list.append(tst_proc_args)
        
    p = Pool(num_processes)
    trn_data_pt_text_list = p.starmap(generate_doc_fts, trn_args_list)
    trn_data_pt_texts = np.vstack(trn_data_pt_text_list)
    
    tst_data_pt_text_list = p.starmap(generate_doc_fts, tst_args_list)
    tst_data_pt_texts = np.vstack(tst_data_pt_text_list)
    
    save_numpy_arrs(
        [
           trn_data_pt_texts, tst_data_pt_texts, label_texts 
        ],
        [
            'trn_data_pt_texts', 'tst_data_pt_texts', 'label_texts'
        ]
    )
    
    print_stats(trn_X_Y, tst_X_Y, trn_data_pt_texts, tst_data_pt_texts, label_texts)
    max_len = max(doc_text_seq_len, label_text_seq_len)
    label_text_extended = np.zeros((label_texts.shape[0], max_len), dtype=np.int64)
    label_text_extended[:, :label_text_seq_len] = label_texts
    sp.save_npz(f'{dataset_dir}/trn_X_Y.npz', trn_X_Y)
    sp.save_npz(f'{dataset_dir}/tst_X_Y.npz', tst_X_Y)
    trn_doc_am, tst_doc_am, lbl_am = get_attention_masks()
    save_numpy_arrs_as_memmap(
        [
            trn_data_pt_texts, tst_data_pt_texts, label_text_extended,
            trn_doc_am, tst_doc_am, lbl_am
        ],
        [
            'trn_doc_input_ids.dat', 'tst_doc_input_ids.dat', 'lbl_input_ids.dat',
            'trn_doc_attention_mask.dat', 'tst_doc_attention_mask.dat', 'lbl_attention_mask.dat'
        ]
    )

generate_data()

    
