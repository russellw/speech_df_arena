import pandas as pd
import numpy as np

def compute_det_curve(target_scores, nontarget_scores):
    """ frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      frr:         np.array,  false rejection rates measured at multiple thresholds
      far:         np.array,  false acceptance rates measured at multiple thresholds
      thresholds:  np.array,  thresholds used to compute frr and far

    frr, far, thresholds have same shape = len(target_scores) + len(nontarget_scores) + 1
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    # false acceptance rates
    # print(float(all_scores[indices[0]]) - 0.001)
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    # Thresholds are the sorted scores
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ eer, eer_threshold = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      eer:              scalar,  value of EER
      eer_threshold:    scalar,  value of threshold corresponding to EER
    """

    frr, far, thresholds = compute_det_curve(
        np.array(target_scores).astype(np.longdouble),
        np.array(nontarget_scores).astype(np.longdouble),
    )
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_eer_API(score_file, protocol_file):
    """eer = compute_eer_API(score_file, protocol_file)
    
    input
    -----
      score_file:     string, path to the socre file
      protocol_file:  string, path to the protocol file
    
    output
    ------
      eer:  scalar, eer value
      
    The way to load text files using read_csv depends on the text format.
    Please change the read_csv if necessary
    """
    protocol_df = pd.read_csv(
            protocol_file,
            names=["file_name", "label"],
            index_col="file_name",
        )

    # load score
    score_df = pd.read_csv(
        score_file,
        names=["file_name", "cm_score"],
        index_col="file_name",
        skipinitialspace=True,
        sep= " ",
        header=0,
    )
    merged_pd = score_df.join(protocol_df)

    bonafide_scores = merged_pd.query('label == "bonafide"')["cm_score"].to_numpy()
        
    spoof_scores = merged_pd.query('label == "spoof"')["cm_score"].to_numpy()
    eer, th = compute_eer(bonafide_scores, spoof_scores)
    return eer, th
